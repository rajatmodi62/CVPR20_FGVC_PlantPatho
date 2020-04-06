from blessed import Terminal

term = Terminal()


def cprint(*args, type="info1"):
    if type == "warn":
        print(term.red(*args))
    if type == "info1":
        print(term.gold(*args))
    elif type == "info2":
        print(term.yellow(*args))
    elif type == "info3":
        print(term.deepskyblue(*args))
    elif type == 'success':
        print(term.lawngreen(*args))
