import cntk as C

def deserialize(func, ctf_path, model, randomize=True, repeat=True, is_test=False):
    """
    Read ctf format to minibatch source, input function
    """
    if not is_test:
        mb_source = C.io.MinibatchSource(
            C.io.CTFDeserializer(
                ctf_path,
                C.io.StreamDefs(
                    c1 = C.io.StreamDef('c1', shape=model.word_dim, is_sparse=True),
                    c2 = C.io.StreamDef('c2', shape=model.word_dim, is_sparse=True),
                    y  = C.io.StreamDef('y', shape=1, is_sparse=False))),
            randomize=randomize,
            max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

        input_map = {
            argument_by_name(func, 'c1'): mb_source.streams.c1,
            argument_by_name(func, 'c2'): mb_source.streams.c2,
            argument_by_name(func, 'y'): mb_source.streams.y
        }
    else:
        mb_source = C.io.MinibatchSource(
            C.io.CTFDeserializer(
                ctf_path,
                C.io.StreamDefs(
                    c1 = C.io.StreamDef('c1', shape=model.word_dim, is_sparse=True),
                    c2 = C.io.StreamDef('c2', shape=model.word_dim, is_sparse=True))),
            randomize=randomize,
            max_sweeps=C.io.INFINITELY_REPEAT if repeat else 1)

        input_map = {
            argument_by_name(func, 'c1'): mb_source.streams.c1,
            argument_by_name(func, 'c2'): mb_source.streams.c2
        }    
    return mb_source, input_map

def argument_by_name(func, name):
    """
    Helper function used to map the variable
    """
    found = [arg for arg in func.arguments if arg.name == name]
    if len(found) == 0:
        raise ValueError('no matching names in arguments')
    elif len(found) > 1:
        raise ValueError('multiple matching names in arguments')
    else:
        return found[0]
