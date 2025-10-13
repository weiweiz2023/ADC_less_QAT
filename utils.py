import time

class Timer:
    def __init__(self, total):
        self.total = total
        self.accum = 0
        self.tpass = 0

    def __enter__(self):
        self.start = time.time()
        print(f"Beginning timer with {self.total} total iterations")
        self.print()
        return self

    def __exit__(self, *args):
        build_str = "".join(['#'] * 10)
        build_str = '[' + build_str + ']' + f"   Iter {self.accum}/{self.total}"
        print(build_str)
        print(f"Execution took {self.tpass:.2f} seconds\n\n")

    def update(self):
        self.accum += 1
        self.tpass += time.time() - self.start
        self.start = time.time()
        self.print()

    def print(self):
        frac = int((self.accum / self.total) * 100)
        hashes = frac // 10
        slashes = (frac % 10) // 5
        build_str = "".join(['#'] * hashes + ['/']*slashes + ['-']*(10-slashes-hashes))
        if self.accum == 0:
            time_str = "??"
        else:
            time_str = f"{(self.tpass / self.accum):.2f}"
        build_str = '\r[' + build_str + ']' + f"   Iter {self.accum}/{self.total}   [ {time_str}s/it ]"
        print(build_str, end='')
        
        
if __name__ == '__main__':
    i = 47
    j = 89
    with Timer(i * j) as timer:
        for i1 in range(i):
            for j1 in range(j):
                time.sleep(0.34)
                timer.update()