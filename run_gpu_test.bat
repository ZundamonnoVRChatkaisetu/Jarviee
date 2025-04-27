@echo off
REM Jarviee GPU�e�X�g�o�b�`
REM GPU�̓���m�F�p�o�b�`�BGPU�̓���󋵂����O�ɋL�^���܂��B

REM ���������擾���ă��O�t�@�C�����ɗ��p
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set LOGFILE=logs\gpu_test_%datetime:~0,8%_%datetime:~8,6%.log

REM ���O�f�B���N�g�����Ȃ���΍쐬
if not exist logs mkdir logs

REM ���O�t�@�C���̃w�b�_�[�o��
echo ===== Jarviee GPU�e�X�g�J�n ===== > %LOGFILE%
echo ���s����: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo ===== Jarviee GPU�e�X�g�J�n =====
echo ���ʂ� %LOGFILE% �ɋL�^����܂�

REM ���ϐ��̐ݒ�
set USE_GPU=true
set GPU_LAYERS=-1
set JARVIEE_DEBUG=1
echo ���ϐ��ݒ�: >> %LOGFILE%
echo USE_GPU=%USE_GPU% >> %LOGFILE%
echo GPU_LAYERS=%GPU_LAYERS% >> %LOGFILE%
echo JARVIEE_DEBUG=%JARVIEE_DEBUG% >> %LOGFILE%
echo. >> %LOGFILE%

REM CUDA���̎擾
echo CUDA���: >> %LOGFILE%
nvidia-smi >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM GPU�ˑ���Python�p�b�P�[�W�̃C���X�g�[��
echo GPU�ˑ���Python�p�b�P�[�W�̃C���X�g�[�����J�n���܂�...
echo ===== GPU�ˑ���Python�p�b�P�[�W�̃C���X�g�[�� ===== >> %LOGFILE%
python scripts\install_gpu_deps.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM GPU�̓���m�F
echo GPU�̓���m�F���s���܂�...
echo ===== GPU�̓���m�F ===== >> %LOGFILE%
python scripts\diagnose_jarviee.py >> %LOGFILE% 2>&1
echo. >> %LOGFILE%

REM ���s���邩���[�U�[�Ɋm�F
echo ���s���܂����H(y/n)
set /p choice=
echo �I����: %choice% >> %LOGFILE%

if /i "%choice%"=="y" (
    echo ���s���܂�...
    echo ===== ���s���܂� ===== >> %LOGFILE%
    set CUDA_VISIBLE_DEVICES=0
    set USE_GPU=true
    set GPU_LAYERS=-1
    echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES% >> %LOGFILE%
    
    echo �����J�n: %time% >> %LOGFILE%
    python -c "import sys; sys.path.append('.'); from src.core.llm.providers.gemma_provider import GemmaProvider; provider = GemmaProvider('./models/gemma-3-12B-it-QAT-Q4_0.gguf', {'use_gpu': True, 'n_gpu_layers': -1, 'verbose': True}); result = provider.generate('こ�?��?�?を英語に翻訳してください: こんにちは、世界?���??気ですか?�?'); print(f'生�?�結果: {result}')" >> %LOGFILE% 2>&1
    echo �����I��: %time% >> %LOGFILE%
)

echo. >> %LOGFILE%
echo �����I��: %date% %time% >> %LOGFILE%
echo ========================================== >> %LOGFILE%

echo �������ʂ� %LOGFILE% �ɋL�^����Ă��܂��B���s����ɂ͉����L�[�������Ă�������...
pause
