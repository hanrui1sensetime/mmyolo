_base_ = ['./base_dynamic.py']
onnx_config = dict(
    input_shape=(640, 640),
    output_names=[
        'cls_scores_0', 'cls_scores_1', 'cls_scores_2', 'bbox_preds_0',
        'bbox_preds_1', 'bbox_preds_2', 'centernesses_0', 'centernesses_1',
        'centernesses_2'
    ])
codebase_config = dict(mode='tensor')
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=True, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 640, 640],
                    opt_shape=[8, 3, 640, 640],
                    max_shape=[32, 3, 640, 640])))
    ])
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501
