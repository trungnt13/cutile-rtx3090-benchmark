import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    ct.store(c, index=(pid,), tile=a_tile + b_tile)


def main() -> None:
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = cp.random.uniform(-1.0, 1.0, vector_size).astype(cp.float32)
    b = cp.random.uniform(-1.0, 1.0, vector_size).astype(cp.float32)
    c = cp.zeros_like(a)

    ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (a, b, c, tile_size))
    cp.cuda.get_current_stream().synchronize()

    np.testing.assert_allclose(cp.asnumpy(c), cp.asnumpy(a) + cp.asnumpy(b), rtol=1e-5, atol=1e-5)
    print("cuTile hello world OK on", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())


if __name__ == "__main__":
    main()
