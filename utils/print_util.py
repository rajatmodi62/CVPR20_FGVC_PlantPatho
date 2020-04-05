from blessed import Terminal

term = Terminal()

def cprint( *args, type=None ):
    if type=="warn":
        print( term.red( *args ) )
    if type=="info1":
        print( term.gold( *args ) )
    elif type == "info2":
        print( term.yellow( *args ) )
    elif type == 'success':
        print( term.lawngreen( *args ) )
    else:
        print( *args )
        