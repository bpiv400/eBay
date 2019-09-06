"""
File giving constants used to update time valued features after some
event has occurred

"""
# event triggered by a buyer or seller accepting an offer from their counter party
SALE = 'SALE'
# event triggered by a slr rejecting an offer on turn 2 or turn 4
SLR_REJECTION = 'SLR_REJECTION'
# event triggered by slr rejecting an offer on the last turn
SLR_REJECTION_FINAL = 'SLR_REJECTION_FINAL' # not used for the timebeing
# event triggered by a buyer rejecting a seller's offer -- closes the thread
BYR_REJECTION = 'BYR_REJECTION'
# event triggered by a buyer or seller offer that's not an acceptance or rejection
OFFER = 'OFFER'
LSTG_EXPIRATION = 'LSTG_EXPIRATION'
# if allowing an offer to expire is equivalent to a rejection, we no longer need this
# THREAD_EXPIRATION = 'THREAD_EXPIRATION'
# new thread is unnecessary when lstgs are independent
# NEW_THREAD = 'NEW_THREAD'
