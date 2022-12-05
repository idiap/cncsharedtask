#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: Copyright © <2022> Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Martin Docekal <idocekal@fit.vutbr.cz>
# SPDX-FileContributor: Martin Fajcik <ifajcik@fit.vutbr.cz>
#
# SPDX-License-Identifier: MIT-License

from collections import Counter
from typing import Collection, List, Optional, Tuple


def get_bestF1_span(
    passage: List, answer: List
) -> Tuple[Optional[Tuple[int, int]], float]:
    """
    Returns best f1 match for span within a passage with the provided answer.
    :param passage: Passage that will be searched for a span with best match.
    :type passage: Collection
    :param answer: An answer that we are trying to match.
    :type answer: Collection
    :return: Span with biggest f1 score represent by a tuple with start and end offsets. Also its f1 score.
        When the span is None it means that there are no shared tokens (f1 score = 0.0).
    :rtype: Tuple[Optional[Tuple[int, int]], float]
    """

    # Let's say that one wants to ask a question how much it is still profitable to increase length of investigated
    # spans if I already investigated all spans of certain len L and I have the best f1 for tham already?
    # Because the f1 score can be expressed by:
    #   f1=2*S/(PT+GT)
    #       (S - shared tokens, PT - number of predicted span tokens, GT - number of ground truth span tokens)
    # We can be optimistic and state that if we will increase the size of predicted span we will increase the
    # number of shared tokens. The maximum number of shared tokens is S=GT. So from the expression for f1
    # and the fact that max S is GT we can get this parametrized (by x) upperbound:
    #   2*GT/(x*GT+GT) .
    # Where we expressed the PT as PT=x*GT. This function has a maximum when x=1 (f1=1) and also we must state that
    # x>=1 (for the length increase, this is what we do, x>1), because we can not have PT < GT and at the same time
    # get S=GT.
    # Ok, so now we need to get the value of parameter x somehow. We will get it with following inequality that
    # symbolically represents the profit condition:
    #   2*SK/(L+GT) < 2*GT/(x*GT+GT)
    # Where the SK is the already known number of shared tokens for spans of length L.
    # Next we can express the x from this inequality:
    #   2*SK/(L+GT) < 2*GT/(x*GT+GT)
    #   SK/(L+GT) < GT/((x+1)GT)
    #   SK/(L+GT) < 1/(x+1)
    #   (x+1) < (L+GT)/SK
    #   x < (L+GT)/SK - 1
    #   x < (L+GT-SK)/SK
    # So now we know that the x must be in (1, (L+GT-SK)/SK), therefore the PT length can be investigated maximally
    # to the length PT < GT*(L+GT-SK)/SK
    #
    # Also there is worth to mention that when the SK=0 we can omit the search, because we didn't find any
    # common token for any span at size L. The expression:
    #   x < (L+GT-SK)/SK ,
    # in limit says, that we can search indefinitely :).
    #

    bestSpan = None  # Tuple (start, end) both indices will be inclusive
    bestSpanScore = 0.0

    # At this time we know nothing like Jon Snow.
    # So we start with 2 and because we are starting our search from spans with 1 tokens only two cases may occur:
    #   We do not find any match so we will just do single iteration and we are done with nothing found.
    #   We found a match and than we update the upper bound online.
    spanLenUpperBound = 2

    actSpanSize = 1

    answerCounter = Counter(answer)
    while actSpanSize < spanLenUpperBound:
        for x in range(len(passage)):
            endOffset = x + actSpanSize
            if endOffset > len(passage):
                break
            spanTokens = passage[x:endOffset]
            score = f1Score(spanTokens, answer)
            if score > bestSpanScore:
                bestSpan = (x, endOffset - 1)  # -1 we want inclusive indices
                bestSpanScore = score

                # let's update the upper bound
                common = Counter(spanTokens) & answerCounter
                common = sum(common.values())
                L = endOffset - x
                spanLenUpperBound = min(
                    len(answer) * (L + len(answer) - common) / common, len(passage) + 1
                )

        actSpanSize += 1

    return bestSpan, bestSpanScore


def f1Score(predictionTokens: Collection, groundTruthTokens: Collection) -> float:
    """
    Calculates f1 score for tokens. The f1 score can be used as similarity measure between output and
    desired output.
    This score is symmetric.
    :param predictionTokens:
    :type predictionTokens: Collection
    :param groundTruthTokens:
    :type groundTruthTokens: Collection
    :return: f1 score
    :rtype: float
    """
    common = Counter(predictionTokens) & Counter(groundTruthTokens)
    numSame = sum(common.values())  # number of true positives
    if numSame == 0:
        # will guard the division by zero
        return 0

    # derivation of used formula:
    #   precision = numSame / len(predictionTokens) = n / p
    #   recall = numSame / len(groundTruthTokens) = n / g
    #   f1 = (2*precision*recall) / (precision+recall) = ((2*n*n)/(p*g)) / (n/p+n/g)
    #   = ((2*n*n)/(p*g)) / ((n*g+n*p)/p*g) = (2*n*n) / (n*g+n*p) = 2*n / (p+g)
    return (2.0 * numSame) / (len(predictionTokens) + len(groundTruthTokens))
