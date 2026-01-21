from scheduler import MicroAssembler
from problem import HASH_STAGES, VLEN, SLOT_LIMITS, SCRATCH_SIZE

class TempPool:
    def __init__(self, asm, size):
        self.asm = asm
        self.size = size
        self.sets = []
        for i in range(size):
            self.sets.append(self._alloc_set(i))

    def _alloc_set(self, i):
        # Alloc temps needed for one vector iteration
        s = {}
        s['scalar_indices'] = [self.asm.alloc(f"s_idx_{i}_{j}") for j in range(VLEN)]
        s['node_vals_v'] = self.asm.alloc(f"node_vals_v_{i}", VLEN)
        s['tmp1_v'] = self.asm.alloc(f"tmp1_v_{i}", VLEN)
        s['tmp2_v'] = self.asm.alloc(f"tmp2_v_{i}", VLEN)
        s['tmp_mod'] = self.asm.alloc(f"tmp_mod_{i}", VLEN)
        s['cond_v'] = self.asm.alloc(f"cond_v_{i}", VLEN)
        s['step_v'] = self.asm.alloc(f"step_v_{i}", VLEN)
        s['mux_temps'] = [self.asm.alloc(f"mux_tmp_{i}_{k}", VLEN) for k in range(4)]
        return s

    def get(self, idx):
        return self.sets[idx % self.size]

class OptimizedKernelBuilder:
    def __init__(self):
        print("DEBUG: OptimizedKernelBuilder initialized!")
        self.asm = MicroAssembler()
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}

    def debug_info(self):
        from problem import DebugInfo
        debug_map = {}
        for name, addr in self.asm.scratch_map.items():
            debug_map[addr] = (name, 1)
        return DebugInfo(scratch_map=debug_map)

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        asm = self.asm

        rounds_p = asm.alloc("rounds")
        n_nodes_p = asm.alloc("n_nodes")
        batch_size_p = asm.alloc("batch_size")
        forest_p = asm.alloc("forest_p")
        inp_idx_p = asm.alloc("inp_idx_p")
        inp_val_p = asm.alloc("inp_val_p")

        tmp = asm.alloc("tmp")
        for i, dest in enumerate([rounds_p, n_nodes_p, batch_size_p, None, forest_p, inp_idx_p, inp_val_p]):
             if dest is not None:
                 asm.emit("load", ("const", tmp, i))
                 asm.emit("load", ("load", dest, tmp))

        N_VECS = batch_size // VLEN
        vec_indices = [asm.alloc(f"v_idx_{i}", VLEN) for i in range(N_VECS)]
        vec_values = [asm.alloc(f"v_val_{i}", VLEN) for i in range(N_VECS)]

        addr_reg = asm.alloc("addr_reg")
        offset_reg = asm.alloc("offset_reg")

        for i in range(N_VECS):
            asm.emit("load", ("const", offset_reg, i * VLEN))
            asm.emit("alu", ("+", addr_reg, inp_idx_p, offset_reg))
            asm.emit("load", ("vload", vec_indices[i], addr_reg))

        for i in range(N_VECS):
            asm.emit("load", ("const", offset_reg, i * VLEN))
            asm.emit("alu", ("+", addr_reg, inp_val_p, offset_reg))
            asm.emit("load", ("vload", vec_values[i], addr_reg))

        hash_consts = []
        for i, stage in enumerate(HASH_STAGES):
            val1 = stage[1]
            c1_s = asm.alloc(f"h{i}_c1_s")
            c1_v = asm.alloc(f"h{i}_c1_v", VLEN)
            asm.emit("load", ("const", c1_s, val1))
            asm.emit("valu", ("vbroadcast", c1_v, c1_s))

            c3_s = asm.alloc(f"h{i}_c3_s")
            c3_v = asm.alloc(f"h{i}_c3_v", VLEN)
            asm.emit("load", ("const", c3_s, stage[4]))
            asm.emit("valu", ("vbroadcast", c3_v, c3_s))
            hash_consts.append((c1_v, c3_v))

        zero_s = asm.alloc("zero_s")
        asm.emit("load", ("const", zero_s, 0))
        zero_v = asm.alloc("zero_v", VLEN)
        asm.emit("valu", ("vbroadcast", zero_v, zero_s))

        one_s = asm.alloc("one_s")
        asm.emit("load", ("const", one_s, 1))
        one_v = asm.alloc("one_v", VLEN)
        asm.emit("valu", ("vbroadcast", one_v, one_s))

        two_s = asm.alloc("two_s")
        asm.emit("load", ("const", two_s, 2))
        two_v = asm.alloc("two_v", VLEN)
        asm.emit("valu", ("vbroadcast", two_v, two_s))

        n_nodes_v = asm.alloc("n_nodes_v", VLEN)
        asm.emit("valu", ("vbroadcast", n_nodes_v, n_nodes_p))

        # Pre-load tree values
        MAX_MUX_ROUND = 4
        tree_vals = []

        idx_reg = asm.alloc("idx_reg")
        val_reg = asm.alloc("val_reg")

        current_node = 0
        for h in range(MAX_MUX_ROUND):
            count = 2**h
            level_regs = []
            for i in range(count):
                asm.emit("load", ("const", idx_reg, current_node + i))
                asm.emit("alu", ("+", idx_reg, forest_p, idx_reg))
                asm.emit("load", ("load", val_reg, idx_reg))
                v_reg = asm.alloc(f"tree_{h}_{i}", VLEN)
                asm.emit("valu", ("vbroadcast", v_reg, val_reg))
                level_regs.append(v_reg)
            tree_vals.append(level_regs)
            current_node += count

        # Offsets
        round_offsets = []
        for h in range(MAX_MUX_ROUND):
            off_s = asm.alloc(f"roff_{h}_s")
            off_v = asm.alloc(f"roff_{h}_v", VLEN)
            offset_val = 2**h - 1
            asm.emit("load", ("const", off_s, offset_val))
            asm.emit("valu", ("vbroadcast", off_v, off_s))
            round_offsets.append(off_v)

        # Bits for mask gen
        round_bits = []
        for h in range(MAX_MUX_ROUND):
            bits_list = []
            for b in range(h):
                bit_s = asm.alloc(f"bit_{h}_{b}_s")
                bit_v = asm.alloc(f"bit_{h}_{b}_v", VLEN)
                asm.emit("load", ("const", bit_s, b))
                asm.emit("valu", ("vbroadcast", bit_v, bit_s))
                bits_list.append(bit_v)
            round_bits.append(bits_list)

        pool = TempPool(asm, 4)

        asm.emit("flow", ("pause",))

        for r in range(rounds):
            level = r % (forest_height + 1)
            is_mux = level < MAX_MUX_ROUND

            for v_i in range(N_VECS):
                temps = pool.get(v_i)
                scalar_indices = temps['scalar_indices']
                target_base = temps['node_vals_v']
                tmp1_v = temps['tmp1_v']
                tmp2_v = temps['tmp2_v']
                tmp_mod = temps['tmp_mod']
                cond_v = temps['cond_v']
                step_v = temps['step_v']
                mux_temps = temps['mux_temps']

                v_idx = vec_indices[v_i]

                if is_mux:
                    level_vals = tree_vals[level]
                    offset_v = round_offsets[level]

                    local_idx_v = tmp2_v
                    asm.emit("valu", ("-", local_idx_v, v_idx, offset_v))

                    current_layer = level_vals

                    for bit in range(level):
                        bit_v = round_bits[level][bit]

                        shifted = tmp1_v
                        asm.emit("valu", (">>", shifted, local_idx_v, bit_v))

                        mask = cond_v
                        asm.emit("valu", ("&", mask, shifted, one_v))

                        # Use multiply_add for Mux
                        # res = mask * (val1 - val0) + val0
                        next_layer = []
                        count = len(current_layer)
                        for k in range(count // 2):
                            val0 = current_layer[2*k]
                            val1 = current_layer[2*k+1]
                            dest = mux_temps[k]

                            diff = step_v # Reuse step_v as temp
                            asm.emit("valu", ("-", diff, val1, val0))
                            asm.emit("valu", ("multiply_add", dest, mask, diff, val0))
                            next_layer.append(dest)

                        current_layer = next_layer

                    if level == 0:
                        asm.emit("valu", ("+", target_base, level_vals[0], zero_v))
                    else:
                        asm.emit("valu", ("+", target_base, current_layer[0], zero_v))

                else: # Gather
                    for j in range(VLEN):
                        asm.emit("alu", ("+", scalar_indices[j], forest_p, v_idx + j))
                    for j in range(VLEN):
                         asm.emit("load", ("load", target_base + j, scalar_indices[j]))

                v_val = vec_values[v_i]
                v_node = target_base
                asm.emit("valu", ("^", v_val, v_val, v_node))

                for stage_i, stage in enumerate(HASH_STAGES):
                    op1, op2, op3 = stage[0], stage[2], stage[3]
                    c1, c3 = hash_consts[stage_i]
                    asm.emit("valu", (op1, tmp1_v, v_val, c1))
                    asm.emit("valu", (op3, tmp2_v, v_val, c3))
                    asm.emit("valu", (op2, v_val, tmp1_v, tmp2_v))

                v_idx = vec_indices[v_i]
                # Step Optimization: step = 1 + (val & 1)
                # If val even (lsb 0): 1 + 0 = 1.
                # If val odd (lsb 1): 1 + 1 = 2.

                lsb = tmp_mod # Reuse
                asm.emit("valu", ("&", lsb, v_val, one_v))
                asm.emit("valu", ("+", step_v, one_v, lsb))

                asm.emit("valu", ("*", v_idx, v_idx, two_v))
                asm.emit("valu", ("+", v_idx, v_idx, step_v))

                asm.emit("valu", ("<", cond_v, v_idx, n_nodes_v))
                asm.emit("flow", ("vselect", v_idx, cond_v, v_idx, zero_v))

            if r == rounds - 1:
                pass
            else:
                asm.emit("flow", ("pause",))
                continue

            asm.emit("flow", ("pause",))

        for i in range(N_VECS):
            asm.emit("load", ("const", offset_reg, i * VLEN))
            asm.emit("alu", ("+", addr_reg, inp_idx_p, offset_reg))
            asm.emit("store", ("vstore", addr_reg, vec_indices[i]))

        for i in range(N_VECS):
            asm.emit("load", ("const", offset_reg, i * VLEN))
            asm.emit("alu", ("+", addr_reg, inp_val_p, offset_reg))
            asm.emit("store", ("vstore", addr_reg, vec_values[i]))

        self.instrs = asm.finish()
