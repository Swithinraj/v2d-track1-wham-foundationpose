from analysis.standings import rank_sums, overall_table


def test_equal_rank_sums_have_stable_order_and_equal_overall_rank():
    table={'h':{'zeta':1.,'alpha':2.,'winner':0.},'o':{'zeta':2.,'alpha':1.,'winner':0.}}
    _,totals=rank_sums(table)
    assert totals==[('winner',2),('alpha',5),('zeta',5)]
    assert overall_table(totals)==[(1,'winner',2),(2,'alpha',5),(2,'zeta',5)]
