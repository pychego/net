import torch
import numpy as np
from model import FC
from dataset import DealDataset
import matplotlib.pyplot as plt
import shap

ckpt = "ckpt/fc-0304.ckpt"
model = FC(17, 5)
ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
model.load_state_dict(ckpt)
model.eval()
test_data = np.load("test.npy")
test_data = DealDataset(test_data)
# print(test_data[325])
data = test_data[:100]
data = torch.from_numpy(data).float()
explainer = shap.DeepExplainer(model, data)
row_for_explain = test_data[100:150]
row_for_explain = torch.from_numpy(row_for_explain).float()
shap_values = explainer.shap_values(row_for_explain)
print(len(shap_values), len(explainer.expected_value))
# shap.initjs()
# print(shap_values[1].shape, explainer.expected_value)
shap.force_plot(explainer.expected_value[0], shap_values[0], row_for_explain, matplotlib=True)
plt.show()