# Deine gobal varibales here 
import fyersSession as fy

def init(isTesting):
    global globalOptionList
    global fyers
    global testing

    globalOptionList=[]
    fyers=fy.getFyerSession()
    testing=True if isTesting else False
