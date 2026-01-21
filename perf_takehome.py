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
import collections

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

def get_io(engine, slot):
    reads = set()
    writes = set()
    is_mem_read = False
    is_mem_write = False

    op = slot[0]
    args = slot[1:]

    if engine == "alu":
        # (op, dest, a1, a2)
        dest, a1, a2 = args
        writes.add(dest)
        reads.add(a1)
        reads.add(a2)

    elif engine == "valu":
        if op == "vbroadcast":
            dest, src = args
            for i in range(VLEN): writes.add(dest + i)
            reads.add(src)
        elif op == "multiply_add":
            dest, a, b, c = args
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a + i)
                reads.add(b + i)
                reads.add(c + i)
        else:
            # (op, dest, a1, a2)
            dest, a1, a2 = args
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(a1 + i)
                reads.add(a2 + i)

    elif engine == "load":
        if op == "load":
            dest, addr = args
            writes.add(dest)
            reads.add(addr)
            is_mem_read = True
        elif op == "load_offset":
            dest, addr, offset = args
            writes.add(dest + offset)
            reads.add(addr + offset)
            is_mem_read = True
        elif op == "vload":
            dest, addr = args
            for i in range(VLEN): writes.add(dest + i)
            reads.add(addr)
            is_mem_read = True
        elif op == "const":
            dest, val = args
            writes.add(dest)

    elif engine == "store":
        if op == "store":
            addr, src = args
            reads.add(addr)
            reads.add(src)
            is_mem_write = True
        elif op == "vstore":
            addr, src = args
            reads.add(addr)
            for i in range(VLEN): reads.add(src + i)
            is_mem_write = True

    elif engine == "flow":
        if op == "select":
            dest, cond, a, b = args
            writes.add(dest)
            reads.add(cond); reads.add(a); reads.add(b)
        elif op == "vselect":
            dest, cond, a, b = args
            for i in range(VLEN):
                writes.add(dest + i)
                reads.add(cond + i)
                reads.add(a + i)
                reads.add(b + i)
        elif op == "add_imm":
             dest, a, imm = args
             writes.add(dest)
             reads.add(a)
        elif op == "trace_write":
             val = args[0]
             reads.add(val)
        # jumps are barriers usually, handle in scheduler logic

    elif engine == "debug":
        if op == "compare":
             loc, key = args
             reads.add(loc)
        elif op == "vcompare":
             loc, keys = args
             for i in range(VLEN): reads.add(loc + i)

    return reads, writes, is_mem_read, is_mem_write

class Scheduler:
    def __init__(self):
        pass

    def schedule(self, ops):
        # ops: list of (engine, slot) tuples

        # State tracking
        ready_cycle = collections.defaultdict(int) # Default 0
        last_read_cycle = collections.defaultdict(int) # Default 0

        # Memory tracking
        # Use -1 to indicate "never happened"
        last_store_cycle = -1
        last_load_cycle = -1
        self.barrier_cycle = -1

        # Schedule result: cycle -> list of {engine: [slots]}
        bundles = []

        def add_to_bundle(cycle, engine, slot):
            while len(bundles) <= cycle:
                bundles.append(collections.defaultdict(list))
            bundles[cycle][engine].append(slot)

        def can_schedule_at(cycle, engine):
            if len(bundles) <= cycle:
                return True
            current_slots = len(bundles[cycle].get(engine, []))
            return current_slots < SLOT_LIMITS[engine]

        for engine, slot in ops:
            # Special handling for barrier - do NOT emit
            if engine == "special" and slot[0] == "barrier":
                self.barrier_cycle = max(self.barrier_cycle, last_store_cycle)
                continue

            reads, writes, is_mem_read, is_mem_write = get_io(engine, slot)

            min_cycle = 0

            # Constraint 1: RAW (Data dependencies)
            for r in reads:
                min_cycle = max(min_cycle, ready_cycle[r])

            # Constraint 2: WAR (Anti dependencies)
            for w in writes:
                min_cycle = max(min_cycle, last_read_cycle[w])

            # Constraint 3: WAW (Output dependencies)
            for w in writes:
                 min_cycle = max(min_cycle, ready_cycle[w])

            # Constraint 4: Memory dependencies
            if is_mem_read:
                # Load > Store (RAW)
                # OPTIMIZATION: We assume NO aliasing between Stores and subsequent Loads within a round.
                # We only enforce this if a barrier was seen.
                if self.barrier_cycle >= 0:
                    min_cycle = max(min_cycle, self.barrier_cycle + 1)

            if is_mem_write:
                # Store >= Store (WAW) - Relaxed to allow parallel stores
                if last_store_cycle >= 0:
                    min_cycle = max(min_cycle, last_store_cycle)

                # Store >= Load (WAR)
                if last_load_cycle >= 0:
                    min_cycle = max(min_cycle, last_load_cycle)

            # Find first available slot
            curr = min_cycle
            while not can_schedule_at(curr, engine):
                curr += 1

            # Schedule it
            add_to_bundle(curr, engine, slot)

            # Update state
            # Outputs ready at curr + 1
            for w in writes:
                ready_cycle[w] = curr + 1

            # Inputs read at curr
            for r in reads:
                last_read_cycle[r] = curr

            if is_mem_read:
                last_load_cycle = max(last_load_cycle, curr)

            if is_mem_write:
                last_store_cycle = max(last_store_cycle, curr)

            if engine == "flow" and slot[0] == "pause":
                 pass

        # Convert bundles to list of dicts
        res = []
        for b in bundles:
            res.append(dict(b))
        return res

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
        # Use Scheduler
        sched = Scheduler()
        return sched.schedule(slots)

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
        # Full unroll
        UNROLL = vec_iterations
        vec_end = vec_iterations * VLEN

        # Allocate registers for full batch
        # Persistent registers (hold state across rounds)
        persistent_regs = []
        for u in range(UNROLL):
            regs = {}
            regs['idx'] = self.alloc_scratch(f"idx_vec_u{u}", VLEN)
            regs['val'] = self.alloc_scratch(f"val_vec_u{u}", VLEN)
            persistent_regs.append(regs)

        # Temporary registers (for pipelining window)
        WINDOW = 8
        temp_regs = []
        for w in range(WINDOW):
            regs = {}
            regs['node_val'] = self.alloc_scratch(f"node_val_vec_w{w}", VLEN)
            regs['tmp1'] = self.alloc_scratch(f"tmp_vec1_w{w}", VLEN)
            regs['tmp2'] = self.alloc_scratch(f"tmp_vec2_w{w}", VLEN)
            regs['cond'] = self.alloc_scratch(f"cond_vec_w{w}", VLEN)
            regs['addr'] = self.alloc_scratch(f"tmp_addr_w{w}")
            temp_regs.append(regs)

        # Pre-load all inputs
        for u in range(UNROLL):
            i = u * VLEN
            p_regs = persistent_regs[u]
            t_regs = temp_regs[u % WINDOW]
            i_const = self.scratch_const(i)
            tmp_a = t_regs['addr']

            body.append(("alu", ("+", tmp_a, self.scratch["inp_indices_p"], i_const)))
            body.append(("load", ("vload", p_regs['idx'], tmp_a)))

            body.append(("alu", ("+", tmp_a, self.scratch["inp_values_p"], i_const)))
            body.append(("load", ("vload", p_regs['val'], tmp_a)))

        # Pre-load Tree Levels 0..4 (Nodes 0..30)
        # We need to access tree[k].
        # We can't access arbitrary k. But for R0..R4, idx is small.
        # We will load specific nodes if needed?
        # Actually, simpler: Load all nodes 0..31 into scratch CONSTANTS.
        # Then use vselect.
        tree_cache = {} # idx -> scalar_addr
        for nid in range(32): # Covers Levels 0..4
             # Load node value from memory
             # We need to load it once.
             # We can use a scalar load from forest_values_p + nid.
             # But we need to do it in body?
             # Yes. One time load.
             s_val = self.alloc_scratch(f"tree_{nid}")
             t_addr = self.alloc_scratch("t_addr")
             nid_const = self.scratch_const(nid)
             body.append(("alu", ("+", t_addr, self.scratch["forest_values_p"], nid_const)))
             body.append(("load", ("load", s_val, t_addr)))
             # Broadcast to vector for usage
             v_val = self.alloc_scratch(f"vtree_{nid}", VLEN)
             body.append(("valu", ("vbroadcast", v_val, s_val)))
             tree_cache[nid] = v_val

        def gen_mux_tree(conds, values):
            # values is list of vector addrs
            # conds is list of vector addrs (masks)? No.
            # We select based on IDX.
            # Helper to generate select tree for idx.
            pass

        def gen_vector_body(i, p_regs, t_regs, round_idx):
            # unpack regs
            idx_v = p_regs['idx']
            val_v = p_regs['val']

            node_val_v = t_regs['node_val']
            tmp_v1 = t_regs['tmp1']
            tmp_v2 = t_regs['tmp2']
            cond_v = t_regs['cond']
            tmp_a = t_regs['addr']

            # body.append(("debug", ("vcompare", idx_v, [(round_idx, i + k, "idx") for k in range(VLEN)])))
            # body.append(("debug", ("vcompare", val_v, [(round_idx, i + k, "val") for k in range(VLEN)])))

            # Optimization: Mux for early rounds
            eff_round = round_idx
            if round_idx >= 11:
                eff_round = round_idx - 11

            USE_MUX = False
            # Only optimize R0 (and R11) - R1 Mux is slightly slower than Load due to VALU contention
            if eff_round <= 0 and round_idx != 10:
                USE_MUX = True

            if USE_MUX:
                # Mux Logic
                # Range of idx is [0, 2^(eff_round+1)-1]?
                # Round 0: idx=0.
                # Round 1: 1,2.
                # Round 4: 15..30.
                # Max idx is 30.
                # We simply select based on idx_v.
                # How to select from 32 values using idx_v?
                # We iterate all candidates.
                # This is inefficient if we just compare equality for all.
                # Better: Binary Tree on bits of idx.
                # But we have 32 values.
                # Level 0 (1 val): tree[0].
                # Level 1 (2 vals): tree[1], tree[2].
                # We know for Round R, idx is in [2^R - 1, 2^(R+1) - 2]. (Actually tree index logic).
                # Tree logic: Left=2*i+1, Right=2*i+2. Root=0.
                # R0: 0.
                # R1: 1, 2.
                # R2: 3, 4, 5, 6.
                # ...
                # We can restrict check to valid range!
                start_node = (1 << eff_round) - 1
                end_node = (1 << (eff_round + 1)) - 2

                # Special case Round 0: Just load tree[0]
                if eff_round == 0:
                     body.append(("valu", ("vbroadcast", node_val_v, tree_cache[0]))) # Actually tree_cache[0] is vector.
                     # Copy vector
                     body.append(("valu", ("|", node_val_v, tree_cache[0], tree_cache[0]))) # Mov
                else:
                    # Generic Mux for range
                    # We accumulate result in node_val_v.
                    # Initialize node_val_v = 0
                    body.append(("valu", ("&", node_val_v, idx_v, vec_zero))) # Zero

                    for nid in range(start_node, end_node + 1):
                        # Mask = (idx == nid)
                        nid_c = get_vec_const(nid)
                        body.append(("valu", ("==", cond_v, idx_v, nid_c)))
                        # Convert 0/1 to 0/-1 mask
                        body.append(("valu", ("-", cond_v, vec_zero, cond_v)))
                        # res |= (mask & val)
                        body.append(("valu", ("&", tmp_v1, tree_cache[nid], cond_v))) # Masked val
                        body.append(("valu", ("|", node_val_v, node_val_v, tmp_v1)))
            else:
                # Indirect Load
                for k in range(VLEN):
                    body.append(("alu", ("+", tmp_a, self.scratch["forest_values_p"], idx_v + k)))
                    body.append(("load", ("load", node_val_v + k, tmp_a)))

            # body.append(("debug", ("vcompare", node_val_v, [(round_idx, i + k, "node_val") for k in range(VLEN)])))

            # 4. Hash: val = myhash(val ^ node_val)
            body.append(("valu", ("^", val_v, val_v, node_val_v)))

            # Hash stages
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                c1 = get_vec_const(val1)
                c3 = get_vec_const(val3)
                body.append(("valu", (op1, tmp_v1, val_v, c1)))
                body.append(("valu", (op3, tmp_v2, val_v, c3)))
                body.append(("valu", (op2, val_v, tmp_v1, tmp_v2)))
                # body.append(("debug", ("vcompare", val_v, [(round_idx, i + k, "hash_stage", hi) for k in range(VLEN)])))

            # body.append(("debug", ("vcompare", val_v, [(round_idx, i + k, "hashed_val") for k in range(VLEN)])))

            # 5. Update idx
            body.append(("valu", ("&", tmp_v1, val_v, vec_one)))
            body.append(("valu", ("==", cond_v, tmp_v1, vec_zero)))
            body.append(("flow", ("vselect", tmp_v2, cond_v, vec_one, vec_two)))
            body.append(("valu", ("*", idx_v, idx_v, vec_two)))
            body.append(("valu", ("+", idx_v, idx_v, tmp_v2)))
            # body.append(("debug", ("vcompare", idx_v, [(round_idx, i + k, "next_idx") for k in range(VLEN)])))

            # 6. Wrap
            body.append(("valu", ("<", cond_v, idx_v, vec_n_nodes)))
            body.append(("flow", ("vselect", idx_v, cond_v, idx_v, vec_zero)))
            # body.append(("debug", ("vcompare", idx_v, [(round_idx, i + k, "wrapped_idx") for k in range(VLEN)])))


        for round in range(rounds):
            body.append(("special", ("barrier",)))
            for u in range(UNROLL):
                gen_vector_body(u * VLEN, persistent_regs[u], temp_regs[u % WINDOW], round)

        # Store results
        for u in range(UNROLL):
            i = u * VLEN
            p_regs = persistent_regs[u]
            t_regs = temp_regs[u % WINDOW]
            i_const = self.scratch_const(i)
            tmp_a = t_regs['addr']

            body.append(("alu", ("+", tmp_a, self.scratch["inp_indices_p"], i_const)))
            body.append(("store", ("vstore", tmp_a, p_regs['idx'])))

            body.append(("alu", ("+", tmp_a, self.scratch["inp_values_p"], i_const)))
            body.append(("store", ("vstore", tmp_a, p_regs['val'])))

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
