from mmdet.apis import init_detector
import torch
import onnx
import warnings
from torch import nn
import onnxruntime
import numpy as np

class stage1Model(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    
    def forward(self,data1):
        x = self.model.extract_feat(data1)    #list
        rpn_outs = self.model.rpn_head(x)    #list
        # proposal_list = get_bboxes(rpn_outs) #[1,1000,5]
        return rpn_outs,x

class stage2Model(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model
    
    def forward(self,data2):
        cls_score, bbox_pred = self.model.roi_head.bbox_head(data2)
        return cls_score, bbox_pred

def pytorch2onnx(model,
                 data,
                 output_file='tmp.onnx',
                 opset_version=11,
                 dynamic_axes=None,
                 output_names=None,
                 show=False,
                 do_simplify=True,
                 ):   
        
    torch.onnx.export(
            model,
            data,
            output_file,
            input_names=["input"],
            output_names=output_names,
            verbose=show,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version)

    if do_simplify:
        import onnxsim
        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            warnings.warn('Failed to simplify ONNX model.')
    print(f'Successfully exported ONNX model: {output_file}')


def val1(data1,model1,save_model1_path):
    onnx_model = onnxruntime.InferenceSession(
            save_model1_path, providers=["CPUExecutionProvider"]
        )
    ort_inputs = {onnx_model.get_inputs()[0].name: data1.numpy()}
    ort_outs = onnx_model.run(None, ort_inputs)
    torch_outs,x = model1(data1)
    for i in range(5):
        print(np.allclose(ort_outs[i],torch_outs[0][i].detach().numpy(),atol=1e-4))
    for i in range(5,10):
        print(np.allclose(ort_outs[i],torch_outs[1][i-5].detach().numpy(),atol=1e-4))
    for i in range(10,15):
        print(np.allclose(ort_outs[i],x[i-10].detach().numpy(),atol=1e-4))


def val2(data2,model2,save_model2_path):

    onnx_model = onnxruntime.InferenceSession(
            save_model2_path, providers=["CPUExecutionProvider"]
        )
    ort_inputs = {onnx_model.get_inputs()[0].name: data2.numpy()}
    ort_outs = onnx_model.run(None, ort_inputs)
    torch_outs = model2(data2)
    
    for i in range(2):
        print(np.allclose(torch_outs[i].detach().numpy(),ort_outs[i],atol=1e-4))
    
    
    
    
    
if __name__=='__main__':
    

    
    save_model1_path = "ckpt/model1.onnx"
    save_model2_path = "ckpt/model2.onnx"
    data1 = torch.rand(1,3,800,1344)
    data2 = torch.rand(1000,256,7,7)

    config_file = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = 'ckpt/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
    model = init_detector(config_file, checkpoint_file,device='cpu')


    model1 = stage1Model(model)
    model2 = stage2Model(model)
    
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
    output_names=['output']

    pytorch2onnx(model1,data1,save_model1_path)
    pytorch2onnx(model2,data2,save_model2_path,dynamic_axes=dynamic_axes,output_names=output_names)

    val1(data1,model1,save_model1_path)
    val2(data2,model2,save_model1_path)
    
    