"""Small synthetic dataset for checking the toolchain; not robot evaluation."""
import numpy as np
from PIL import Image, ImageDraw
from .recorder import EpisodeRecorder


def _observation(index):
    image = Image.new('RGB', (256, 192), '#102436')
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 130, 244, 172), fill='#566575')
    draw.rectangle((150, 110, 215, 133), fill='#38c5b2')
    x = min(170, 35 + 7 * index)
    draw.ellipse((x, 95, x+20, 115), fill='#ffb84d')
    draw.text((14, 12), f'SYNTHETIC DEMO / step {index}', fill='white')
    return {'agentview_image': np.array(image),
        'robot0_eye_in_hand_image': np.array(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        'robot0_eef_pos': np.array([x / 256, .1, .2]),
        'robot0_gripper_qpos': np.array([.02, .02])}


def create_demo(root):
    recorder = EpisodeRecorder(root, {'task_description': 'Synthetic: move cup onto plate',
        'synthetic': True, 'seed': 0, 'raw_image_orientation': 'upright_demo'})
    for index in range(24):
        recorder.append(_observation(index), np.array([.02,0,0,0,0,0,1.]), _observation(index+1),
            reward=float(index == 23), terminated=index==23,
            context={'phase': 'policy', 'chunk_id': index//8, 'chunk_index': index%8,
                'predicates_before': {'on(cup, plate)': index >= 20},
                'predicates_after': {'on(cup, plate)': index+1 >= 20},
                'success_after': index==23, 'source': 'synthetic_demo'})
    recorder.finish(outcome='success')
    return recorder.path
