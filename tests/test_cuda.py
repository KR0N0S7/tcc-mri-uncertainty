import torch

print('=== Teste CUDA ===')
print(f'PyTorch:           {torch.__version__}')
print(f'CUDA disponível:   {torch.cuda.is_available()}')

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'Device:            {torch.cuda.get_device_name(0)}')
    print(f'VRAM total:        {props.total_memory/1e9:.2f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
    print(f'CUDA runtime:      {torch.version.cuda}')

    x = torch.randn(2000, 2000).cuda()
    y = x @ x.T
    torch.cuda.synchronize()
    print(f'\nMatmul 2000x2000 na GPU: OK (output shape {y.shape})')
    print(f'Pico de VRAM no teste: {torch.cuda.max_memory_allocated()/1e6:.1f} MB')
else:
    print('CUDA NÃO está disponível — algo deu errado na instalação')

print('\n=== Teste fastmri ===')
import fastmri
from fastmri.models import VarNet
print(f'fastmri importado OK')
modelo = VarNet(num_cascades=2, chans=8, pools=2)
print(f'VarNet de teste criada com {sum(p.numel() for p in modelo.parameters())/1e6:.2f}M params')