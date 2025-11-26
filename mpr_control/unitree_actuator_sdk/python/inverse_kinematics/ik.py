import math

def compute_ik_3dof(x, y, z, dh):
    """
    Computes the first 3 joint angles (theta1, theta2, theta3) in radians
    for a simplified 3-DOF manipulator with the given DH parameters.
    
    Joints 4,5,6 are assumed zero for this demonstration,
    or you can fix them to some orientation.

    Args:
      x, y, z : end-effector position in meters
      dh      : dict of DH params, e.g. {'d1':0.4, 'a2':0.3, 'a3':0.25, 'd6':0.08}
    
    Returns:
      (theta1, theta2, theta3) in radians
    """
    d1 = dh['d1']  # offset along Z0
    a2 = dh['a2']  # link length from J1 to J2
    a3 = dh['a3']  # link length from J2 to J3
    # d6 = dh['d6']  # if you'd like to subtract wrist extension

    # For a complete 6-DOF, you'd subtract d6 * R_EE*zHat to get wrist center
    # But if we keep orientation = 0, we can ignore for now.

    # Wrist center (naive)
    xWC = x
    yWC = y
    zWC = z

    # 1) theta1 = atan2(y, x)
    theta1 = math.atan2(yWC, xWC)

    # 2) geometry in the r-s plane
    r = math.sqrt(xWC**2 + yWC**2)
    s = zWC - d1

    # 3) Law of cosines for theta3
    #    cos(theta3) = (r^2 + s^2 - a2^2 - a3^2) / (2 * a2 * a3)
    cosT3 = (r*r + s*s - a2*a2 - a3*a3) / (2.0 * a2 * a3)
    # clamp numerical issues
    cosT3 = max(-1.0, min(1.0, cosT3))
    theta3 = math.acos(cosT3)

    # 4) sin part
    sinT3 = math.sin(theta3)

    # 5) theta2 = atan2(s, r) - atan2(a3*sinT3, a2 + a3*cosT3)
    theta2 = math.atan2(s, r) - math.atan2(a3*sinT3, a2 + a3*cosT3)

    return (theta1, theta2, theta3)
