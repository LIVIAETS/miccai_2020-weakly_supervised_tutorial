
data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

results.gif: results/images/TOY/unconstrained results/images/TOY/constrained
	./gifs.sh

results/images/TOY/unconstrained: data/TOY
	python main.py --dataset TOY --mode unconstrained --gpu

results/images/TOY/constrained: data/TOY
	python main.py --dataset TOY --mode constrained --gpu