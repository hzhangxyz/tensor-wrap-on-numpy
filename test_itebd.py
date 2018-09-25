import numpy_wrap as np

a = np.tensor([[[-1,2],[3,-4.]],[[6,-2],[-4,5]]], legs=['pa', 'ra','da'])
b = np.tensor([[[4,-2],[1,-3.]],[[3,-3],[2,5]]], legs=['pb', 'lb','db'])
c = np.tensor([[[-3,-1],[-2,3.]],[[-3,5],[1,-2]]], legs=['pc', 'rc','uc'])
d = np.tensor([[[3,-2],[1,-6.]],[[5,-2],[-3,4]]], legs=['pd', 'ld','ud'])

h = np.tensor(
        np.array([1, 0, 0, 0, 0, -1, 2, 0, 0, 2, -1, 0, 0, 0, 0, 1])
        .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])/4.
Iden = np.tensor(
        np.identity(4)
        .reshape([2, 2, 2, 2]), legs=['p1', 'p2', 'P1', 'P2'])

evol = Iden - 0.001 * h

for t in range(100000):
    ab = a\
            .tensor_contract(b,['ra'],['lb'])\
            .tensor_contract(evol, ['pa','pb'],['p1','p2'],{},{'P1':'pa','P2':'pb'})
    ab /= np.linalg.norm(ab)
    a, s, b = ab.tensor_svd(['pa','da'],['pb','db'],['ra','lb'])
    a = a[:2,:2,:2]
    b = b[:2,:2,:2]
    a.tensor_multiple(s[:2],'ra')

    cd = c\
            .tensor_contract(d,['rc'],['ld'])\
            .tensor_contract(evol, ['pc','pd'],['p1','p2'],{},{'P1':'pc','P2':'pd'})
    cd /= np.linalg.norm(cd)
    c, s, d = cd.tensor_svd(['pc','uc'],['pd','ud'],['rc','ld'])
    c = c[:2,:2,:2]
    d = d[:2,:2,:2]
    c.tensor_multiple(s[:2],'rc')

    """
    ac = a\
            .tensor_contract(c,['da'],['uc'])\
            .tensor_contract(evol, ['pa','pc'],['p1','p2'],{},{'P1':'pa','P2':'pc'})
    ac /= np.linalg.norm(ac)
    a, s, c = ac.tensor_svd(['pa','ra'],['pc','rc'],['da','uc'])
    a = a[:2,:2,:2]
    c = c[:2,:2,:2]
    a.tensor_multiple(s[:2],'da')

    bd = b\
            .tensor_contract(d,['db'],['ud'])\
            .tensor_contract(evol, ['pb','pd'],['p1','p2'],{},{'P1':'pb','P2':'pd'})
    bd /= np.linalg.norm(bd)
    b, s, d = bd.tensor_svd(['pb','lb'],['pd','ld'],['db','ud'])
    b = b[:2,:2,:2]
    d = d[:2,:2,:2]
    b.tensor_multiple(s[:2],'db')
    """

    psi = a\
            .tensor_contract(b,['ra'],['lb'])\
            .tensor_contract(c,['da'],['uc'])\
            .tensor_contract(d,['rc','db'],['ld','ud'])
    Hpsi = np.tensor(np.zeros([2,2,2,2]),legs=['pa','pb','pc','pd'])
    #for i, j in [['pa','pb'],['pb','pd'],['pa','pc'],['pb','pd']]:
    for i, j in [['pa','pb'],['pc','pd']]:
        Hpsi += psi\
            .tensor_contract(h, [i,j],['p1','p2'],{},{'P1':i,'P2':j})

    print(psi.tensor_contract(Hpsi,psi.legs,psi.legs)/psi.tensor_contract(psi,psi.legs,psi.legs)/4)

