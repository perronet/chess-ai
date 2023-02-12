all: play

convert:
	cd engine && python3 convert_dataset.py

train: convert
	cd engine && python3 train.py

play: train
	cd experiments && python3 pygui.py
