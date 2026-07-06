from recorder.main import main


'''
# 确认音频设备编号
$env:PYTHONIOENCODING='utf-8'; python -m sounddevice
# 结果如下
21 外部麦克风 (Realtek(R) Audio), Windows WASAPI (2 in, 0 out)


# 运行命令
python main.py --out_dir data/trial_001 --duration 60 --audio_device 21 --rs_warmup_frames 30
# 不手动设置时间
python main.py --out_dir data/trial_001_test --audio_device 21 --rs_warmup_frames 30
'''
if __name__ == "__main__":
    main()
