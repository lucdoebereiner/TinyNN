// one input, one output

TinyNN11 {
	var weights1, weights2, bias1, bias2,
	z1, a1, z2, a2, x, <>output, error, weights1D, <>loss;

    *new { | size |
        ^super.new.init(size)
    }

    init { | size |
		weights1 = Array.rand(size, -1.0, 1.0);
		weights2 = Array.rand(size, -1.0, 1.0);
		bias1 = Array.rand(size, -1.0, 1.0);
		bias2 = -1.0.rrand(1.0);
    }

	forwardPass { |xIn|
		x = xIn;
		z1 = weights1.collect({ |w, i| ([x] * w).sum + bias1[i]});
		a1 = z1.tanh;
		z2 = (a1 * weights2).sum + bias2;
		a2 = z2.tanh;
		output = a2;
	}

	cost { |y|
		var mean = { |input| input.isArray.if({input.mean}, input) };
		error = mean.((y - a2).squared);
	}

	backprop { |y|
		var tanhD = { |x| 1 - x.tanh.squared };
		// layer 1
		var weights1G = weights2.collect({ |w, i|
			(a2 - y) * tanhD.(z2) * w * tanhD.(z1[i]) * x;
		});
		var bias1G =  weights2.collect({ |w, i|
			(a2 - y) * tanhD.(z2) * w * tanhD.(z1[i]);
		});
		// layer 2
		var weights2G = (a2 - y) * tanhD.(z2) * a1;
		var bias2G = (a2 - y) * tanhD.(z2);
		^[weights1G, bias1G, weights2G, bias2G]
	}

	// todo batch size
	train { |data, lr=0.01, epochs=100, report=true|
		loss = [];
		epochs.do({
			var w1G = [];
			var b1G = [];
			var w2G = [];
			var b2G = [];
			var losses = [];
			data.do({ |d|
				var grads;
				this.forwardPass(d[0]);
				this.cost(d[1]);
				losses = losses.add(error);
				grads = this.backprop(d[1]);
				w1G = w1G.add(grads[0]);
				b1G = b1G.add(grads[1]);
				w2G = w2G.add(grads[2]);
				b2G = b2G.add(grads[3]);
			});
			report.if({"Loss: ".post; losses.mean.postln});
			loss = loss.add(losses.mean);
			weights1 = weights1 - (w1G.mean * lr);
			weights2 = weights2 - (w2G.mean * lr);
			bias1 = bias1 - (b1G.mean * lr);
			bias2 = bias2 - (b2G.mean * lr);
		});
	}

	genAutoreg { |start, fac=1, n = 100|
		var output = [start];
		var x = start;
		n.do({
			this.forwardPass(x*fac);
			x = this.output;
			output = output.add(x);
		})
		^output
	}
}



