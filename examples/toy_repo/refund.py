from coupon import reactivate_coupon

def process_refund(order, coupon):
    order.refunded = True
    # BUG: coupon reactivation is missing
    return order
