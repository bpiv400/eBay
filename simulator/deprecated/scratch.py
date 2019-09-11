    def input_training_data(self, train):
        # save data and parameters
        self.train = train
        self.N = torch.sum(~torch.isnan(train['y'])).item()
        if self.isLSTM:
            self.steps = train['y'].size()[0]

        # for constructing batch indices
        self.v = [i for i in range(train['y'].size()[-1])]
        self.batches = int(np.ceil(len(self.v) / MBSIZE))

        if self.EM:
            if self.omega is None:
                vals = np.full(
                    tuple(train['y'].size()) + (self.K,), 1/self.K)
                self.omega = torch.as_tensor(vals,
                    dtype=torch.float).detach()
            else:
                print('Cannot reset training data for EM models.')


def train_long_model(simulator, x_fixed, data):
    lnL = np.full(EPOCHS, np.nan)

    y = data['y'][simulator.model][simulator.outcome]
    lstgs = np.unique(y.index.get_level_values('lstg').copy())

    for i in range(EPOCHS):
        np.random.shuffle(lstgs)
        indices = np.array_split(lstgs, N_GROUPS)

        lnL_n = np.full(N_GROUPS, np.nan)
        N = np.full(N_GROUPS, np.nan)

        for n in range(N_GROUPS):
            start = dt.now()

            # function to select lstg indices
            parse = lambda df: df[df.index.isin(
                np.sort(indices[n]), level='lstg')].sort_index()

            # create dictionary of parsed dataframes
            d = {}

            # parse fixed features
            d['x_fixed'] = parse(x_fixed)

            # expand outcome vector
            if simulator.outcome == 'days':
                d['y'] = parse_days(parse(y))
            elif simulator.outcome == 'delay':
                d['y'] = parse_delay(simulator.model, parse(y))

            # expand timestep features
            d['x_time'] = parse_time_feats_long(simulator.model,
                d['y'].index, data['z']['start'], parse(data['z'][model]))

            # convert to tensors and input to simulator
            simulator.input_training_data(convert_to_tensors(d))

            # run epoch and record log-likelihood and number of obs
            lnL_n[n] = simulator.run_epoch()
            N[n] = simulator.N

            print('Round %d of %d: %dsec. N: %d. lnL: %1.4f.' %
                (n+1, N_SLR_GROUPS, (dt.now() - start).seconds,
                    N[n], lnL_n[n]))

        # average log-likelihood for epoch
        lnL[i] = np.dot(lnL_n, N) / np.sum(N)

    return lnL

    # return loss history and total duration
    return {'lnL': lnL, 'duration': dt.now() - time0}

def process_inputs(model, outcome, data):
    y, x = [data[k] for k in ['y', 'x']]
    # initialize output dictionaries
    d = {}
    N = {}
    # fill in values given parameters
    if outcome == 'days':
        d['x_fixed'] = x['lstg']
    elif outcome == 'delay':
        d['x_fixed'] = parse_fixed_feats_delay(x, y[model][outcome])
    else:
        d['y'] = y[model][outcome]
        if model == 'arrival':
            d['x_fixed'] = parse_fixed_feats_arrival(outcome, x)
            N['time'] = N_TIME_FEATS
        else:
            d['x_fixed'] = parse_fixed_feats_role(x)
            d['x_time'] = parse_time_feats_role(model, outcome, x['offer'])
            N['time'] = len(d['x_time'].columns)
    N['fixed'] = len(d['x_fixed'].columns)
    return d, N


if __name__ == '__main__':
    # extract parameters from command line
    parser = argparse.ArgumentParser(
        description='Model of environment for bargaining AI.')
    parser.add_argument('--model', type=str,
        help='One of: arrival, byr, slr.')
    parser.add_argument('--outcome', type=str, help='Outcome to predict.')
    parser.add_argument('--id', type=int, help='Experiment ID.')
    args = parser.parse_args()

    # extract parameters from CSV
    params = get_params(args)
    print(params)

    # load data
    print('Loading data')
    #data = pickle.load(open(TRAIN_PATH, 'rb'))
    data = pickle.load(open('../../data/chunks/1_out.pkl', 'rb'))

    # create inputs to model
    train, N = process_inputs(args.model, args.outcome, data)

    # initialize neural net
    simulator = Simulator(args.model, args.outcome, params, N)
    print(simulator.net)

    # train model
    print('Training')
    time0 = dt.now()
    if args.outcome not in ['days', 'delay']:
        simulator.input_training_data(convert_to_tensors(train))
        output = train_model(simulator)
    else:
        output = train_long_model(simulator, train['x_fixed'], data)
    dur = dt.now() - time0

    # save simulator parameters and other output
    prefix = BASEDIR + '%d' % args.id
    torch.save(simulator.net.state_dict(), prefix + '.pt')
    pickle.dump(output, open(prefix + '.pkl', 'wb'))


# nested fixed features for arrival models
    for z in ARRIVAL_MODELS:
        v = y['arrival'][z]
        if z != 'days':
            x_fixed = x_fixed.loc[v.index]
        if outcome == z:
            return x_fixed
        x_fixed = x_fixed.join(v)
        if z == 'hist':
            x_fixed[z] = np.log(1 + x_fixed[z])
        if z == 'bin':
            x_fixed.drop('bin', axis=1, inplace=True)


def parse_theta(theta, K):
    # parameters
    n = theta.size()[-1]
    M = n - 2 * K

    # exponentiate
    theta = torch.exp(theta)

    # parse beta parameters
    a = 1 + torch.index_select(theta, -1, torch.tensor(range(M, K + M)))
    b = 1 + torch.index_select(theta, -1, torch.tensor(range(K + M, n)))

    # discrete probabilities
    if M > 0:
        eta = torch.index_select(theta, -1, torch.tensor(range(M)))
        p = torch.div(eta, 1 + torch.sum(eta, dim=-1, keepdim=True))
    else:
        p = None

    return p, a, b


def beta_mixture_loss(theta, y, gamma):
    p, a, b = parse_theta(theta, gamma.size()[-1])

    # beta densities
    lndens = Beta(a, b).log_prob(y.unsqueeze(dim=-1))

    # calculate negative log-likelihood
    if p is None:
        return -torch.sum(torch.sum(gamma * lndens))
    else:
        idx = (y > 0) & (y < 1) & ~torch.isnan(y)

        Q = torch.sum(gamma[idx] * lndens[idx], dim=1)
        ll = torch.sum(torch.log(1 - torch.sum(p, -1)[idx]) + Q)

        for i in range(2):
            ll += torch.sum(torch.log(p[y == i, [i]]))

        return -ll

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


def get_round(offer):
    digits = np.ceil(np.log10(offer))
    factor = 5 * np.power(10, digits-3)
    rounded = np.round(offer / factor) * factor
    isRound = (rounded == offer).astype(np.float64)
    isRound.loc[np.isnan(offer)] = np.nan
    isNines = ((rounded > offer) &
               (rounded - offer <= factor / 5)).astype(np.float64)
    isNines.loc[np.isnan(offer)] = np.nan
    isNines.loc[isRound == 1.] = np.nan
    return isRound, isNines
