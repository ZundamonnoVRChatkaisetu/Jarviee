@echo off
REM Jarviee GPU�f�f�X�N���v�g
REM GPU�̃T�|�[�g�󋵂��ڍׂɃ`�F�b�N���܂�

echo ===== Jarviee GPU�f�f�X�N���v�g =====

REM Python�̃o�[�W�����m�F
echo Python�o�[�W����:
python --version

REM ���ϐ���ݒ�
set USE_GPU=true
set GPU_LAYERS=-1
set JARVIEE_DEBUG=1

REM NVIDIA-SMI�̎��s
echo.
echo NVIDIA-SMI���:
nvidia-smi

REM PyTorch��CUDA�T�|�[�g�m�F
echo.
echo PyTorch��CUDA�T�|�[�g:
python -c "import torch; print(f'CUDA���p�\: {torch.cuda.is_available()}'); print(f'GPU��: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('GPU��������܂���');"

REM llama-cpp-python��GPU�T�|�[�g�m�F
echo.
echo llama-cpp-python��GPU�T�|�[�g:
python -c "try: from llama_cpp import Llama; import inspect; gpuSupport = 'n_gpu_layers' in inspect.signature(Llama.__init__).parameters; print(f'GPU�Ή���llama-cpp-python: {'�L��' if gpuSupport else '����'}'); except ImportError: print('llama-cpp-python���C���X�g�[������Ă��܂���');"

REM �f�f�c�[���̎��s
echo.
echo GPU�C���X�g�[���[�f�f�c�[�������s���܂����H(y/n)
set /p choice=

if /i "%choice%"=="y" (
    python scripts\install_gpu_support.py --force
)

echo.
echo GPU�T�|�[�g�t���`�F�b�N�����s���܂����H(y/n)
set /p choice=

if /i "%choice%"=="y" (
    python scripts\diagnose_jarviee.py
)

pause
