'''
Utility functions specific to wolfPHC algorithm
'''


'''Get State index from seller actions'''
def allSellerActions2stateIndex(allSellerActions,N,sellerActionSize):
    stateIndex = 0
    for i in range(0,N):
        stateIndex = stateIndex * sellerActionSize + allSellerActions[i]
    return stateIndex