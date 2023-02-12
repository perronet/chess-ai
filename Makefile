all: convert train play

convert:
	cd engine && python3 convert_dataset.py

train:
	cd engine && python3 train.py

play:
	cd experiments && python3 pygui.py
