import numpy as np
import torch
from tensor_node import Node

def main(D=4, t=0.1, N=100, H=torch.Tensor([[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]])/4):
    # initial
    H = H.reshape([2,2,2,2])
    I = torch.Tensor(np.identity(4)).reshape([2,2,2,2])

    A = Node(torch.randn([D,D,2]), "L R P".split(" "))
    B = Node(torch.randn([D,D,2]), "L R P".split(" "))
    EAB = torch.ones([D])
    EBA = torch.ones([D])

    #update
    Up = Node(I - t * H, "P1 P2 P3 P4".split(" "))
    def updateAB():
        nonlocal A, B, EAB, EBA
        U, env, V = A.tensor_multiple(EBA, "L") \
                     .tensor_multiple(EAB, "R") \
                     .rename_legs({"P": "P1"}) \
                     .tensor_contract(B, ["R"], ["L"]) \
                     .tensor_multiple(EBA, "R") \
                     .rename_legs({"P": "P2"}) \
                     .tensor_contract(Up, ["P1", "P2"], ["P3", "P4"]) \
                     .tensor_svd(["L", "P1"], ["R", "P2"], ["R", "L"], D)
        A = U.tensor_multiple(1/EBA, "L").rename_legs({"P1":"P"})
        B = V.tensor_multiple(1/EBA, "R").rename_legs({"P2":"P"})
        EAB = env[:D]
    def updateBA():
        nonlocal A, B, EAB, EBA
        U, env, V = B.tensor_multiple(EAB, "L") \
                     .tensor_multiple(EBA, "R") \
                     .rename_legs({"P": "P1"}) \
                     .tensor_contract(A, ["R"], ["L"]) \
                     .tensor_multiple(EAB, "R") \
                     .rename_legs({"P": "P2"}) \
                     .tensor_contract(Up, ["P1", "P2"], ["P3", "P4"]) \
                     .tensor_svd(["L", "P1"], ["R", "P2"], ["R", "L"], D)
        B = U.tensor_multiple(1/EAB, "L").rename_legs({"P1":"P"})
        A = V.tensor_multiple(1/EAB, "R").rename_legs({"P2":"P"})
        EBA = env[:D]

    for i in range(N):
        print(i)
        updateAB()
        EAB /= torch.max(torch.abs(EAB))
        updateBA()
        EBA /= torch.max(torch.abs(EBA))

    # energy
    Hami = Node(H, "P1 P2 P3 P4".split(" "))
    def energy():
        Left = Node(torch.ones([D,D]),"R1 R2".split())
        Right = Node(torch.ones([D,D]),"L1 L2".split())
        for _ in range(N):
            Left = Left \
                .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1"}) \
                .tensor_contract(A, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
                .tensor_multiple(EAB, "R1") \
                .tensor_multiple(EAB, "R2") \
                .tensor_contract(B, ["R1"], ["L"], {}, {"R":"R1"}) \
                .tensor_contract(B, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
                .tensor_multiple(EBA, "R1") \
                .tensor_multiple(EBA, "R2")
            Left = Left.tensor_scalar(1/torch.max(torch.abs(Left.data)))
            Right = Right \
                .tensor_contract(A, ["L1"], ["R"], {}, {"L":"L1"}) \
                .tensor_contract(A, ["L2", "P"], ["R", "P"], {}, {"L":"L2"}) \
                .tensor_multiple(EBA, "L1") \
                .tensor_multiple(EBA, "L2") \
                .tensor_contract(B, ["L1"], ["R"], {}, {"L":"L1"}) \
                .tensor_contract(B, ["L2", "P"], ["R", "P"], {}, {"L":"L2"}) \
                .tensor_multiple(EAB, "L1") \
                .tensor_multiple(EAB, "L2")
            Right = Right.tensor_scalar(1/torch.max(torch.abs(Right.data)))
        psipsi = Left \
            .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1"}) \
            .tensor_contract(A, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
            .tensor_multiple(EAB, "R1") \
            .tensor_multiple(EAB, "R2") \
            .tensor_contract(B, ["R1"], ["L"], {}, {"R":"R1"}) \
            .tensor_contract(B, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
            .tensor_multiple(EBA, "R1") \
            .tensor_multiple(EBA, "R2") \
            .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1"}) \
            .tensor_contract(A, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
            .tensor_contract(Right, ["R1", "R2"], ["L1", "L2"])
        psiHABpsi = Left \
            .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1", "P":"P1"}) \
            .tensor_contract(A, ["R2"], ["L"], {}, {"R":"R2", "P":"P3"}) \
            .tensor_multiple(EAB, "R1") \
            .tensor_multiple(EAB, "R2") \
            .tensor_contract(B, ["R1"], ["L"], {}, {"R":"R1", "P":"P2"}) \
            .tensor_contract(B, ["R2"], ["L"], {}, {"R":"R2", "P":"P4"}) \
            .tensor_contract(Hami, "P1 P2 P3 P4".split(" "), "P1 P2 P3 P4".split()) \
            .tensor_multiple(EBA, "R1") \
            .tensor_multiple(EBA, "R2") \
            .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1"}) \
            .tensor_contract(A, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
            .tensor_contract(Right, ["R1", "R2"], ["L1", "L2"])
        psiHBApsi = Left \
            .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1"}) \
            .tensor_contract(A, ["R2", "P"], ["L", "P"], {}, {"R":"R2"}) \
            .tensor_multiple(EAB, "R1") \
            .tensor_multiple(EAB, "R2") \
            .tensor_contract(B, ["R1"], ["L"], {}, {"R":"R1", "P":"P1"}) \
            .tensor_contract(B, ["R2"], ["L"], {}, {"R":"R2", "P":"P3"}) \
            .tensor_multiple(EBA, "R1") \
            .tensor_multiple(EBA, "R2") \
            .tensor_contract(A, ["R1"], ["L"], {}, {"R":"R1", "P":"P2"}) \
            .tensor_contract(A, ["R2"], ["L"], {}, {"R":"R2", "P":"P4"}) \
            .tensor_contract(Hami, "P1 P2 P3 P4".split(" "), "P1 P2 P3 P4".split()) \
            .tensor_contract(Right, ["R1", "R2"], ["L1", "L2"])
        return (psiHABpsi.data + psiHBApsi.data)/psipsi.data/2
    print(energy())

if __name__ == "__main__":
    from sys import argv
    D = int(argv[1])
    t = float(argv[2])
    N = int(argv[3])
    main(D=D, t=t, N=N)
