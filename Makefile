


results.gif: results/images/TOY/unconstrained results/images/TOY/constrained
	./gifs.sh

results/images/TOY/unconstrained: data/TOY
	python main.py --dataset TOY --mode unconstrained --gpu

results/images/TOY/constrained: data/TOY
	python main.py --dataset TOY --mode constrained --gpu