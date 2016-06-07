# do this:
"""
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates
    """
#store cleaned-up, padded(?) lines in a 2-d list in memory (line index, words)
#import updated vecs as a 2-d list (line index, value)
def update_key(d, params, batch_of_lines, updated_vecs):
    list_of_words = []
    for line in batch_of_lines:
        if len(line) < params['MAX_LENGTH']:
            #pad(line)
        list_of_words.extend(line)
    while list_of_words.count('') > 0:
        index = list_of_words.index('')
        del list_of_words[index]
        del updated_vecs[index]
    #2 lists:
    #vectors
    #words
    #find duplicate words, avg vectors
    #what if 2 words are updated differently? do we average??
    for word in list_of_words:
        d[word] = list_of_vectors
    return d
    #takes updated vectors from minibatch
    #updates key

    #uses key for next minibatch/eval
