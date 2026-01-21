from problem import SLOT_LIMITS, VLEN, HASH_STAGES, SCRATCH_SIZE
import heapq

class Scheduler:
    def __init__(self):
        self.ops = [] # List of (id, engine, slot_tuple, dependencies)
        self.next_id = 0
        self.reg_writers = {} # reg -> op_id
        self.reg_readers = {} # reg -> list of op_id

    def add(self, engine, slot, reads, writes):
        op_id = self.next_id
        self.next_id += 1

        deps = set()
        # Read-After-Write (RAW): Depends on previous writer
        for r in reads:
            if r in self.reg_writers:
                deps.add(self.reg_writers[r])
            # Track this op as a reader
            if r not in self.reg_readers: self.reg_readers[r] = []
            self.reg_readers[r].append(op_id)

        # Write-After-Write (WAW) and Write-After-Read (WAR)
        for w in writes:
            # WAW: Depends on previous writer
            if w in self.reg_writers:
                deps.add(self.reg_writers[w])
            # WAR: Depends on previous readers
            if w in self.reg_readers:
                for reader_id in self.reg_readers[w]:
                    if reader_id != op_id:
                        deps.add(reader_id)

            # Update writer
            self.reg_writers[w] = op_id
            # Reset readers (since new value is written)
            self.reg_readers[w] = []

        self.ops.append({
            "id": op_id,
            "engine": engine,
            "slot": slot,
            "deps": deps,
            "succs": [],
            "in_degree": 0
        })
        return op_id

    def schedule(self):
        # Build graph
        ops_map = {op["id"]: op for op in self.ops}
        for op in self.ops:
            for dep_id in op["deps"]:
                ops_map[dep_id]["succs"].append(op["id"])

        # Initial in-degree
        for op in self.ops:
            op["in_degree"] = len(op["deps"])

        ready_queue = [op for op in self.ops if op["in_degree"] == 0]
        # Priority: heuristics
        # ready_queue.sort(...) ?

        # Op completion events: (finish_cycle, op_id)
        # Latency = 1 cycle. If issued at T, finishes at T+1.
        # Successors ready at T+1.
        events = []

        schedule = []
        current_cycle = 0
        executed_count = 0
        total_ops = len(self.ops)

        # Track active slots?
        # We process cycle by cycle.

        while executed_count < total_ops:
            # Process completions
            # Any op finishing at current_cycle or earlier?
            # Latency 1 means if issued at T, ready at T+1.
            # So at start of T, we process ops issued at T-1.
            # (Which finished at T).

            # Actually, `ready_queue` contains ops whose deps are SATISFIED.
            # If dep issued at T-1, it is done at T.
            # So its successors decrement in-degree.
            # If in-degree 0, add to ready_queue.

            # Events logic:
            while events and events[0][0] <= current_cycle:
                _, finished_op_id = heapq.heappop(events)
                finished_op = ops_map[finished_op_id]
                for succ_id in finished_op["succs"]:
                    succ = ops_map[succ_id]
                    succ["in_degree"] -= 1
                    if succ["in_degree"] == 0:
                        ready_queue.append(succ)

            # Issue from ready_queue
            # Sort by heuristic?
            # Critical path: long term.
            # Engine availability: short term.
            # Simple heuristic: Number of successors.
            ready_queue.sort(key=lambda x: len(x["succs"]), reverse=True)

            slots_used = {e: 0 for e in SLOT_LIMITS}
            bundle = {e: [] for e in SLOT_LIMITS}

            issued_indices = []

            for i, op in enumerate(ready_queue):
                eng = op["engine"]
                if slots_used[eng] < SLOT_LIMITS[eng]:
                    slots_used[eng] += 1
                    bundle[eng].append(op["slot"])
                    issued_indices.append(i)
                    heapq.heappush(events, (current_cycle + 1, op["id"]))
                    executed_count += 1

            # Remove issued from ready_queue (in reverse index order)
            for i in sorted(issued_indices, reverse=True):
                ready_queue.pop(i)

            schedule.append(bundle)
            current_cycle += 1

            if not issued_indices and executed_count < total_ops and not events:
                # Deadlock?
                raise RuntimeError("Scheduler deadlock!")

            if not issued_indices and executed_count < total_ops:
                # Stall (waiting for events)
                # Advance time to next event
                next_event_time = events[0][0]
                # Add empty bundles for idle cycles
                stall_cycles = next_event_time - current_cycle
                for _ in range(stall_cycles):
                    schedule.append({e: [] for e in SLOT_LIMITS})
                current_cycle = next_event_time

        with open("/tmp/scheduler_stats.txt", "w") as f:
            f.write(f"Scheduler finished in {current_cycle} cycles. Ops: {total_ops}. IPC: {total_ops/current_cycle if current_cycle > 0 else 0:.2f}\n")

        return schedule

class MicroAssembler:
    def __init__(self):
        self.scheduler = Scheduler()
        self.scratch_map = {}
        self.next_scratch = 0

    def alloc(self, name=None, size=1):
        addr = self.next_scratch
        self.next_scratch += size
        if self.next_scratch > SCRATCH_SIZE:
             raise RuntimeError(f"Out of scratch space! Requested {self.next_scratch}")
        if name:
            self.scratch_map[name] = addr
        return addr

    def get_addr(self, name):
        return self.scratch_map[name]

    def emit(self, engine, slot):
        reads = []
        writes = []

        op = slot[0]
        args = slot[1:]

        if engine == "alu":
            writes.append(args[0])
            reads.append(args[1])
            if len(args) > 2: reads.append(args[2])

        elif engine == "valu":
            if op == "vbroadcast":
                dest = args[0]
                src = args[1]
                for i in range(VLEN): writes.append(dest + i)
                reads.append(src)

            elif op == "multiply_add":
                dest, a, b, c = args
                for i in range(VLEN):
                    writes.append(dest + i)
                    reads.append(a + i)
                    reads.append(b + i)
                    reads.append(c + i)
            else:
                dest, a1, a2 = args
                for i in range(VLEN):
                    writes.append(dest + i)
                    reads.append(a1 + i)
                    reads.append(a2 + i)

        elif engine == "load":
            if op == "load":
                writes.append(args[0])
                reads.append(args[1])
            elif op == "load_offset":
                 dest, addr, off = args
                 writes.append(dest+off)
                 reads.append(addr+off)
            elif op == "vload":
                dest, addr = args
                for i in range(VLEN): writes.append(dest+i)
                reads.append(addr)
            elif op == "const":
                writes.append(args[0])

        elif engine == "store":
            if op == "store":
                reads.append(args[0])
                reads.append(args[1])
            elif op == "vstore":
                addr, src = args
                reads.append(addr)
                for i in range(VLEN): reads.append(src+i)

        elif engine == "flow":
            if op == "select":
                dest, cond, a, b = args
                writes.append(dest)
                reads.extend([cond, a, b])
            elif op == "vselect":
                dest, cond, a, b = args
                for i in range(VLEN):
                    writes.append(dest+i)
                    reads.extend([cond+i, a+i, b+i])
            elif op == "add_imm":
                dest, a, imm = args
                writes.append(dest)
                reads.append(a)

        self.scheduler.add(engine, slot, reads, writes)

    def finish(self):
        return self.scheduler.schedule()
