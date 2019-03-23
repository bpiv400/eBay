def get_digit_indicators(offer, digit, prefix):
    '''
    Creates a dataframe with indicators for the following:
        a. offer == digit in hundredths
        b. offer == digit in tenths
        c. offer == digit in ones
        d. offer == digit in tens
        e. offer == digit in hundreds
    '''
    strings = 'XXXXX' + offer.map('{0:.2f}'.format)
    df = pd.DataFrame(index=strings.index)
    indices = [-1, -2, -4, -5, -6]
    for i in range(len(indices)):
        # define place
        newname = '_'.join([prefix, 'digit', string.ascii_lowercase[i] + digit])
        df[newname] = strings.str[indices[i]] == digit
    return df


def compute_example(p, a, b, gamma, simulator):
    model = simulator.get_model()
    if model == 'delay':
        p = torch.mean(p, dim=1).squeeze().detach().numpy()
        Q = torch.sum(torch.mul(gamma, a / (a+b)), dim=2).detach().numpy()
        E_beta = np.nanmean(Q, axis=1)
        for i in range(3):
            print('\tTurn %d: p_exp = %.2f, E_beta = %1.2f hours.' %
                (i+1, p[i], 48 * E_beta[i]))
    elif model == 'msg':
        p = np.mean(theta / (1 + theta), axis=1)
        print('\tTurn 1: %.2f. Turn 2: %.2f. Turn 3: %.2f.' % tuple(p))


def print_summary(epoch, start, lnL_train, simulator):
    sec = (dt.now() - start).seconds
    print('Epoch %d: %dsec. Train lnL: %1.4f. Test lnL: %1.4f.' %
        (epoch + 1, sec, lnL_train, lnL_test))
    #compute_example(p, a, b, gamma, simulator)
    sys.stdout.flush()
