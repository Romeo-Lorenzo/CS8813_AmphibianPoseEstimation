import sleap_io as sio
import os

packages = [
    'dataset/shrimp/propershrimp/shrimp_train.pkg.slp',
    'dataset/shrimp/propershrimp/shrimp_val.pkg.slp',
    'dataset/shrimp/propershrimp/shrimp_test.pkg.slp',
    'dataset/shrimp/Vibha/ApplyingTrainedModelThreeBigShrimp.pkg.slp',
    'dataset/shrimp/Vibha/ShrimpInferedGT.pkg.slp',
    'SLEAP/Shrimps/ApplyingTrainedModelThreeBigShrimp.pkg.slp',
]

print("SHRIMP PACKAGE CONTENTS\n" + "="*80)

for pkg_path in packages:
    full_path = os.path.join(os.getcwd(), pkg_path)
    if os.path.exists(full_path):
        try:
            labels = sio.load_file(full_path)
            num_frames = len(labels.labeled_frames)
            num_nodes = len(labels.skeleton.nodes) if labels.skeleton else 0
            
            # Count instances and labeled points
            total_instances = 0
            total_points = 0
            for frame in labels.labeled_frames:
                total_instances += len(frame.instances)
                for inst in frame.instances:
                    total_points += len([point for point in inst.points if point is not None])
            
            print(f'{pkg_path}:')
            print(f'  Labeled frames: {num_frames}')
            print(f'  Total instances labeled: {total_instances}')
            print(f'  Total labeled points: {total_points}')
            print(f'  Skeleton nodes: {num_nodes}')
            if labels.skeleton:
                nodes_str = ", ".join([n.name for n in labels.skeleton.nodes[:5]])
                if num_nodes > 5:
                    nodes_str += f", ... +{num_nodes-5}"
                print(f'  Node names: {nodes_str}')
            print()
        except Exception as e:
            print(f'{pkg_path}:')
            print(f'  ERROR - {str(e)[:100]}')
            print()
    else:
        print(f'{pkg_path}: NOT FOUND')
        print()
