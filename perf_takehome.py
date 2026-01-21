"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation using SIMD instructions.
        """
        # --- Initialization ---
        # Scalars
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]

        # Helper to load scalar constants
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        for v in init_vars:
            self.alloc_scratch(v, 1)  # scalar

        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Vector Scratch Registers
        idx_vec = self.alloc_scratch("idx_vec", VLEN)
        val_vec = self.alloc_scratch("val_vec", VLEN)
        node_val_vec = self.alloc_scratch("node_val_vec", VLEN)

        tmp_vec1 = self.alloc_scratch("tmp_vec1", VLEN)
        tmp_vec2 = self.alloc_scratch("tmp_vec2", VLEN)
        cond_vec = self.alloc_scratch("cond_vec", VLEN)

        tmp_addr = self.alloc_scratch("tmp_addr")  # scalar for address calculation

        # Helper vars for scalar tail handling
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")

        # Scratch used in scalar hash
        tmp_s1 = self.alloc_scratch("tmp_s1")
        tmp_s2 = self.alloc_scratch("tmp_s2")
        tmp_s3 = self.alloc_scratch("tmp_s3")

        # --- Constants Management ---
        vec_const_map = {}

        def get_vec_const(val):
            if val not in vec_const_map:
                # Load scalar const
                s_addr = self.scratch_const(val)
                # Allocate vector
                v_addr = self.alloc_scratch(f"vec_const_{val}", VLEN)
                # Broadcast
                self.add("valu", ("vbroadcast", v_addr, s_addr))
                vec_const_map[val] = v_addr
            return vec_const_map[val]

        # Pre-load common constants
        vec_zero = get_vec_const(0)
        vec_one = get_vec_const(1)
        vec_two = get_vec_const(2)

        scalar_zero = self.scratch_const(0)
        scalar_one = self.scratch_const(1)
        scalar_two = self.scratch_const(2)

        # We also need vector version of n_nodes for wrapping check
        vec_n_nodes = get_vec_const(n_nodes)

        # Pre-load hash constants
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            get_vec_const(val1)
            get_vec_const(val3)
            # Ensure scalar constants are also available
            self.scratch_const(val1)
            self.scratch_const(val3)

        def build_scalar_hash(val_addr, tmp1, tmp2, round, i):
            slots = []
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                slots.append(("alu", (op1, tmp1, val_addr, self.scratch_const(val1))))
                slots.append(("alu", (op3, tmp2, val_addr, self.scratch_const(val3))))
                slots.append(("alu", (op2, val_addr, tmp1, tmp2)))
                slots.append(("debug", ("compare", val_addr, (round, i, "hash_stage", hi))))
            return slots

        # --- Main Loop ---
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        body = []

        # Process vectorized part
        vec_iterations = batch_size // VLEN
        vec_end = vec_iterations * VLEN

        for round in range(rounds):
            # Vectorized Loop
            for i in range(0, vec_end, VLEN):
                # We process items i to i+VLEN-1
                i_const = self.scratch_const(i)

                # 1. Load indices: idx_vec = mem[inp_indices_p + i : ... + VLEN]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("vload", idx_vec, tmp_addr)))
                body.append(("debug", ("vcompare", idx_vec, [(round, i + k, "idx") for k in range(VLEN)])))

                # 2. Load values: val_vec = mem[inp_values_p + i : ... + VLEN]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("vload", val_vec, tmp_addr)))
                body.append(("debug", ("vcompare", val_vec, [(round, i + k, "val") for k in range(VLEN)])))

                # 3. Indirect Load: node_val_vec[k] = mem[forest_values_p + idx_vec[k]]
                # This must be done scalarly for each lane
                for k in range(VLEN):
                    # addr = forest_values_p + idx_vec[k]
                    # idx_vec[k] is at address idx_vec + k
                    body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], idx_vec + k)))
                    body.append(("load", ("load", node_val_vec + k, tmp_addr)))

                body.append(("debug", ("vcompare", node_val_vec, [(round, i + k, "node_val") for k in range(VLEN)])))

                # 4. Hash: val = myhash(val ^ node_val)
                body.append(("valu", ("^", val_vec, val_vec, node_val_vec)))

                # Hash stages
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # tmp1 = a op1 val1
                    # tmp2 = a op3 val3
                    # a = tmp1 op2 tmp2
                    c1 = get_vec_const(val1)
                    c3 = get_vec_const(val3)

                    body.append(("valu", (op1, tmp_vec1, val_vec, c1)))
                    body.append(("valu", (op3, tmp_vec2, val_vec, c3)))
                    body.append(("valu", (op2, val_vec, tmp_vec1, tmp_vec2)))

                    body.append(("debug", ("vcompare", val_vec, [(round, i + k, "hash_stage", hi) for k in range(VLEN)])))

                body.append(("debug", ("vcompare", val_vec, [(round, i + k, "hashed_val") for k in range(VLEN)])))

                # 5. Update idx: idx = 2*idx + (1 if val % 2 == 0 else 2)
                # cond: (val & 1) == 0
                body.append(("valu", ("&", tmp_vec1, val_vec, vec_one)))
                body.append(("valu", ("==", cond_vec, tmp_vec1, vec_zero)))

                # select increment: 1 if cond else 2
                body.append(("flow", ("vselect", tmp_vec2, cond_vec, vec_one, vec_two)))

                # idx = idx * 2
                body.append(("valu", ("*", idx_vec, idx_vec, vec_two)))

                # idx = idx + incr
                body.append(("valu", ("+", idx_vec, idx_vec, tmp_vec2)))

                body.append(("debug", ("vcompare", idx_vec, [(round, i + k, "next_idx") for k in range(VLEN)])))

                # 6. Wrap: idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", cond_vec, idx_vec, vec_n_nodes)))
                # If idx < n_nodes (cond_vec is 1), keep idx. Else 0.
                body.append(("flow", ("vselect", idx_vec, cond_vec, idx_vec, vec_zero)))

                body.append(("debug", ("vcompare", idx_vec, [(round, i + k, "wrapped_idx") for k in range(VLEN)])))

                # 7. Store results
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, idx_vec)))

                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, val_vec)))

            # Scalar cleanup for tail elements
            for i in range(vec_end, batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))

                body.extend(build_scalar_hash(tmp_val, tmp_s1, tmp_s2, round, i))

                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp_s1, tmp_val, scalar_two)))
                body.append(("alu", ("==", tmp_s1, tmp_s1, scalar_zero)))
                body.append(("flow", ("select", tmp_s3, tmp_s1, scalar_one, scalar_two)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, scalar_two)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp_s3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp_s1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp_s1, tmp_idx, scalar_zero)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
