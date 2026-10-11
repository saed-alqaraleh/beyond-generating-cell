
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import core


def toy_state():
    # Two candidate scenes: 0..3 and 4..7.
    candidate_scene = np.asarray([0,0,0,0,1,1,1,1], dtype=np.int8)
    # Six rows; source positions are fixed whether supported or not.
    source_idx = np.asarray([0,1,2,3,4,5], dtype=np.int32)
    supported = np.asarray([1,0,1,0,1,0], dtype=bool)
    valid = np.zeros((6,8), dtype=bool)
    edges = [
        [0,1,4],    # source 0 pinned
        [0,2,5],    # source 1 forbidden
        [2,3,6],    # source 2 pinned
        [0,1,7],    # source 3 forbidden
        [1,4,6],    # source 4 pinned
        [2,4,7],    # source 5 forbidden
    ]
    for r, cells in enumerate(edges):
        valid[r, cells] = True
    row_scene = np.zeros((6,2), dtype=np.int16)
    for s in range(2):
        row_scene[:,s] = valid[:, candidate_scene==s].sum(axis=1)
    st = core.SwitchState.from_valid(
        valid, source_idx, supported, candidate_scene, row_scene
    )
    return st, valid, source_idx, supported, candidate_scene, row_scene


def test_switch_invariants():
    st, valid, source_idx, supported, candidate_scene, row_scene = toy_state()
    rng = np.random.default_rng(12345)
    prop = np.zeros(3, dtype=np.int64)
    acc = np.zeros(3, dtype=np.int64)
    # SCENES constant has 3 entries, toy only uses first two; third edge list is empty
    # so extend state edge arrays to the production scene count.
    while len(st.edge_rows) < len(core.SCENES):
        st.edge_rows.append(np.asarray([], dtype=np.int32))
        st.edge_cols.append(np.asarray([], dtype=np.int32))
    st.run_proposals(500, rng, prop, acc)
    inv = st.invariant_report()
    assert inv["source_membership_mismatches"] == 0
    assert inv["row_scene_degree_cell_mismatches"] == 0
    assert inv["candidate_column_degree_mismatches"] == 0
    assert inv["total_cardinality_query_mismatches"] == 0


def test_deterministic_seed_and_path():
    a, *_ = toy_state()
    b, *_ = toy_state()
    for st in (a,b):
        while len(st.edge_rows) < len(core.SCENES):
            st.edge_rows.append(np.asarray([], dtype=np.int32))
            st.edge_cols.append(np.asarray([], dtype=np.int32))
    rng_a = np.random.default_rng(core.derive_chain_seed(20260905, 0))
    rng_b = np.random.default_rng(core.derive_chain_seed(20260905, 0))
    pa = np.zeros(3, dtype=np.int64); aa = np.zeros(3, dtype=np.int64)
    pb = np.zeros(3, dtype=np.int64); ab = np.zeros(3, dtype=np.int64)
    a.run_proposals(250, rng_a, pa, aa)
    b.run_proposals(250, rng_b, pb, ab)
    for xa, xb in zip(a.edge_cols, b.edge_cols):
        assert np.array_equal(xa, xb)
    assert rng_a.bit_generator.state == rng_b.bit_generator.state


def test_source_supported_recall_monotonicity():
    # Every supported query whose source is top-1 must be a null hit.
    adj = [{0,2}, {3}, {4,6}]
    top10 = np.zeros((2,3,10), dtype=np.int32)
    top10[:,0,:] = np.asarray([0,1,2,3,4,5,6,7,8,9])
    top10[:,1,:] = np.asarray([3,1,2,0,4,5,6,7,8,9])
    top10[:,2,:] = np.asarray([4,1,2,3,0,5,6,7,8,9])
    hits = core.evaluate_null_hits(adj, top10)
    assert hits[:, :, 0].all()


def test_holm():
    p = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    a = core.holm_adjust(p)
    assert len(a) == 6
    assert all(0 <= x <= 1 for x in a)
    order = np.argsort(p)
    vals = np.asarray(a)[order]
    assert np.all(vals[1:] >= vals[:-1])


def test_simultaneous_interval():
    rng = np.random.default_rng(77)
    point = np.asarray([0.1,0.2,0.3,0.4,0.5,0.6])
    boot = point + rng.normal(0,0.01,size=(10000,6))
    lo, hi, q = core.simultaneous_intervals(point, boot)
    assert q > 0
    assert np.all(lo < point)
    assert np.all(hi > point)
    assert np.all((hi-lo) >= 0)



def test_checkpoint_equivalent_trajectory():
    a, valid, source_idx, supported, candidate_scene, row_scene = toy_state()
    b, *_ = toy_state()
    for st in (a,b):
        while len(st.edge_rows) < len(core.SCENES):
            st.edge_rows.append(np.asarray([], dtype=np.int32))
            st.edge_cols.append(np.asarray([], dtype=np.int32))
    seed=core.derive_chain_seed(20260905, 2)
    rng_a=np.random.default_rng(seed)
    rng_b=np.random.default_rng(seed)
    pa=np.zeros(3,dtype=np.int64); aa=np.zeros(3,dtype=np.int64)
    pb=np.zeros(3,dtype=np.int64); ab=np.zeros(3,dtype=np.int64)
    # uninterrupted path
    a.run_proposals(200, rng_a, pa, aa)
    # path with a synthetic checkpoint at 100 proposal transitions
    b.run_proposals(100, rng_b, pb, ab)
    er,ec=b.edge_arrays_copy()
    state_json=rng_b.bit_generator.state
    b2=core.SwitchState.from_edges(
        edge_rows=er, edge_cols=ec, source_idx=source_idx,
        source_supported=supported, candidate_scene_code=candidate_scene,
        initial_col_degrees=valid.sum(axis=0).astype(np.int32),
        initial_row_scene_counts=row_scene, initial_free_adj=b.initial_free_adj,
    )
    rng_b2=np.random.default_rng(); rng_b2.bit_generator.state=state_json
    b2.run_proposals(100, rng_b2, pb, ab)
    for xa,xb in zip(a.edge_cols,b2.edge_cols):
        assert np.array_equal(xa,xb)
    assert rng_a.bit_generator.state == rng_b2.bit_generator.state


def test_chain_seeds_independent_of_schedule():
    ids=[0,1,2,3]
    forward={i:core.derive_chain_seed(20260905,i) for i in ids}
    reverse={i:core.derive_chain_seed(20260905,i) for i in reversed(ids)}
    assert forward == reverse
    assert len(set(forward.values())) == 4

def test_proposal_clock_includes_rejections():
    st, *_ = toy_state()
    while len(st.edge_rows) < len(core.SCENES):
        st.edge_rows.append(np.asarray([], dtype=np.int32))
        st.edge_cols.append(np.asarray([], dtype=np.int32))
    rng = np.random.default_rng(4321)
    proposed = np.zeros(3, dtype=np.int64)
    accepted = np.zeros(3, dtype=np.int64)
    st.run_proposals(1000, rng, proposed, accepted)
    assert int(proposed.sum()) == 1000
    assert 0 < int(accepted.sum()) < 1000  # confirms rejected self-transitions occurred
    inv = st.invariant_report()
    assert inv["source_membership_mismatches"] == 0
    assert inv["candidate_column_degree_mismatches"] == 0


def test_jump_chain_bias_regression_guard():
    # Degree sequence with five feasible 3x3 bipartite graphs.  Under the
    # symmetric edge-pair proposal, state acceptance probabilities are not
    # constant (1/2 vs 2/3). Therefore an accepted-move-only jump chain would
    # not have the uniform stationary distribution.  This guards against
    # reintroducing accepted-move timing.
    import itertools
    rowdeg = (1, 1, 2)
    coldeg = (1, 1, 2)
    edges = [(r,c) for r in range(3) for c in range(3)]
    states=[]
    for comb in itertools.combinations(edges, sum(rowdeg)):
        rd=[0,0,0]; cd=[0,0,0]
        for r,c in comb:
            rd[r]+=1; cd[c]+=1
        if tuple(rd)==rowdeg and tuple(cd)==coldeg:
            states.append(set(comb))
    assert len(states)==5
    rates=[]
    for G in states:
        E=list(G); total=len(E)*(len(E)-1); ok=0
        for i,(r1,c1) in enumerate(E):
            for j,(r2,c2) in enumerate(E):
                if i==j or r1==r2 or c1==c2:
                    continue
                if (r1,c2) in G or (r2,c1) in G:
                    continue
                ok += 1
        rates.append(ok/total)
    assert min(rates) == 0.5
    assert max(rates) == 2/3


def run_all():
    tests = [
        test_switch_invariants,
        test_deterministic_seed_and_path,
        test_source_supported_recall_monotonicity,
        test_holm,
        test_simultaneous_interval,
        test_checkpoint_equivalent_trajectory,
        test_chain_seeds_independent_of_schedule,
        test_proposal_clock_includes_rejections,
        test_jump_chain_bias_regression_guard,
    ]
    for f in tests:
        f()
        print("PASS", f.__name__)
    print("ALL SOURCE-MATCHED NULL UNIT TESTS: PASS")


if __name__ == "__main__":
    run_all()
