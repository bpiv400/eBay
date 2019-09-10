"""
File giving constants used to update time valued features after some
event has occurred

"""
# event triggered by a buyer or seller accepting an offer from their counter party
SALE = 'SALE'
# event triggered by a slr rejecting an offer on turn 2 or turn 4
SLR_REJECTION = 'SLR_REJECTION'
# event triggered by a buyer rejecting a seller's offer -- closes the thread
BYR_REJECTION = 'BYR_REJECTION'
# event triggered by a buyer or seller offer that's not an acceptance or rejection
OFFER = 'OFFER'
# when a listing finally expires
LSTG_EXPIRATION = 'LSTG_EXPIRATION'

