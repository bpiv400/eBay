class BargainerModel:
    def __init__(self, msg=0, con=0, delay=0):
        pass


class BuyerModel(BargainerModel):
    def __init__(self, msg=0, con=0, delay=0, composer=None):
        super(BuyerModel, self).__init__(msg=msg, con=con, delay=delay,
                                         composer=composer)
        pass


class SellerModel(BargainerModel):
    def __init__(self, msg=0, con=0, delay=0, composer=None):
        super(SellerModel, self).__init__(msg=msg, con=con, delay=delay,
                                          composer=composer)
        pass
