def focal_binary_cross_entropy(logit, truth):

    '''This function defines Focal Loss for Binary variables'''

    weight_pos=2
    weight_neg=1
    gamma=2
    margin=0.2
    em = np.exp(margin)

    logit = logit.view(-1)
    truth = truth.view(-1)
    log_pos = -F.logsigmoid( logit)
    log_neg = -F.logsigmoid(-logit)

    log_prob = truth*log_pos + (1-truth)*log_neg
    prob = torch.exp(-log_prob)
    margin = torch.log(em +(1-em)*prob)

    weight = truth*weight_pos + (1-truth)*weight_neg
    loss = margin + weight*(1 - prob) ** gamma * log_prob

    loss = loss.mean()
    return loss