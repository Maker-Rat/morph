from csmt.models.architecture_humdog import PAN_model


def create_model(args, body_parts, joint_parts, datasets, topology_name):
    return PAN_model(args, body_parts, joint_parts, datasets, topology_name)


# Backward-compat alias used by legacy scripts.
def creat_model(args, body_parts, joint_parts, datasets, topology_name):
    return create_model(args, body_parts, joint_parts, datasets, topology_name)
