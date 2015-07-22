def iter_ngram(seq, max_order, min_order=None, sent_start=None, sent_end=None):
    if min_order > max_order:
        raise ValueError("min_order > max_order (%d > %d)" % (min_order, max_order))

    if min_order is None:
        min_order = max_order

    orders = range(min_order, max_order+1)

    it = iter(seq)

    if sent_start is not None:
        buffer = [sent_start]*max_order
    else:
        buffer = []


    last_countdown = None
    while True:
        if last_countdown is None:
            try:
                item = it.next()
            except StopIteration:
                if sent_end is None:
                    break
                else:
                    last_countdown = max_order - 1
                    item = sent_end
        else:
            if last_countdown <= 1:
                break

            item = sent_end
            last_countdown -= 1

        buffer.append(item)

        del buffer[:-max_order]

        for n in orders:
            if len(buffer) < n:
                continue
            yield buffer[-n:]

def iter_ngram_pad(seq, max_order, min_order=None, sent_start=None, sent_end=None, padding=[]):
    if len(padding) < max_order-1:
        raise ValueError("padding must have at least %d items" % (max_order-1))

    offset = len(padding)-max_order
    
    for ngram in iter_ngram(seq, max_order, min_order, sent_start, sent_end):
        n = len(ngram)
        yield ngram+padding[offset+n:]
