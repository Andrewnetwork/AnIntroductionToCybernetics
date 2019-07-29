from functools import reduce

def fn_loss(fn, transitions):
    error = 0
    for i in transitions:
        error += (fn(i[0])-i[1])**2
    return error/len(transitions)


def compose_fn_ls(ls):
    return lambda z: reduce(lambda x, y: y(x), ls, z)


# TODO: Turn into a generator.
def create_fns(lambda_ls, start_range, end_range):
    outLs = []
    for i in range(start_range, end_range):
        for fn in lambda_ls:
            outLs.append((fn[0].format(i), fn[1](i)))
    return outLs


fns = [("{{0}}**{0}", lambda j: lambda n: n**j),
       ("{0}**{{0}}", lambda j: lambda n: j**n),
       ("{{0}}+{0}", lambda j: lambda n: n+j),
       ("{{0}}-{0}", lambda j: lambda n: n-j),
       ("{0}-{{0}}", lambda j: lambda n: j-n),
       ("{{0}}/{0}", lambda j: lambda n: n/j),
       ("{0}/{{0}}", lambda j: lambda n: j/n),
       ("{{0}}*{0}", lambda j: lambda n: n*j),
       ("{0}", lambda j: lambda n: j),
       ("{{0}}", lambda j: lambda n: n)]


def condense_transformation(transformation, start_constant_range=0, end_constant_range=20, composition_depth=2, fn_ls=fns):
    prime_fns = create_fns(fn_ls, start_constant_range, end_constant_range)
    composite_fns = prime_fns

    for i in range(composition_depth):
        fn_ls_next = []
        for (composite_str, composite_fn) in composite_fns:
            for (prime_fn_str, prime_fn) in prime_fns:
                fn_ls_next.append((prime_fn_str.format(composite_str), compose_fn_ls([composite_fn, prime_fn])))
            try:
                loss = fn_loss(composite_fn, transformation)
                if loss == 0:
                    return "n→"+composite_str.format("n")
            except:
                pass
        composite_fns = fn_ls_next

    return "NOT FOUND"
