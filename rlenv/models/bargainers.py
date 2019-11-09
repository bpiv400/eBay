class BargainerModel:
    def __init__(self, msg=0, con=0, delay=0):
        pass


class BuyerModel(BargainerModel):
    def __init__(self, msg=0, con=0, delay=0):
        super(BuyerModel, self).__init__(msg=msg, con=con, delay=delay)
        pass


class SellerModel(BargainerModel):
    def __init__(self, msg=0, con=0, delay=0):
        super(SellerModel, self).__init__(msg=msg, con=con, delay=delay)
        pass
