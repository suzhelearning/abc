import numpy as np

from abc_minimal.flow import flow_interpolate


def test_flow_direction_is_explicit_for_abc_and_spd():
    data = np.asarray([2.0])
    noise = np.asarray([-2.0])
    point_abc, velocity_abc = flow_interpolate(
        data, noise, np.asarray([0.25]), data_at_one=False
    )
    point_spd, velocity_spd = flow_interpolate(
        data, noise, np.asarray([0.25]), data_at_one=True
    )
    np.testing.assert_allclose(point_abc, [1.0])
    np.testing.assert_allclose(velocity_abc, [-4.0])
    np.testing.assert_allclose(point_spd, [-1.0])
    np.testing.assert_allclose(velocity_spd, [4.0])
