import os, math, random, urllib.request
random.seed(42)

# --- Dataset Setup ---
if not os.path.exists('input.txt'):
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

# --- Autograd Class with GELU ---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0
        self._children, self._local_grads = children, local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    
    def gelu(self):
        v = self.data
        out_data = 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v**3)))
        grad = 0.5 * (1.0 + math.tanh(math.sqrt(2.0/math.pi)*(v + 0.044715*v**3)))
        return Value(out_data, (self,), (grad,))

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# --- Model Config ---
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4
head_dim = n_embd // n_head
lora_r = 4
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wq_lora_a'] = matrix(lora_r, n_embd)
    state_dict[f'layer{i}.attn_wq_lora_b'] = matrix(n_embd, lora_r)
    state_dict[f'layer{i}.moe_gate'] = matrix(2, n_embd)
    for e in range(2):
        state_dict[f'layer{i}.expert{e}.fc1'] = matrix(2 * n_embd, n_embd)
        state_dict[f'layer{i}.expert{e}.fc2'] = matrix(n_embd, 2 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

# --- Core Logic ---
def linear(x, w): return [sum((wi * xi for wi, xi in zip(wo, x)), Value(0)) for wo in w]

def lora_linear(x, w, lora_a, lora_b):
    base = linear(x, w)
    lora_path = linear(linear(x, lora_a), lora_b)
    return [b + l * (1/lora_r) for b, l in zip(base, lora_path)]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps, Value(0))
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum((xi * xi for xi in x), Value(0)) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    x = rmsnorm(state_dict['wte'][token_id])
    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q = lora_linear(x, state_dict[f'layer{li}.attn_wq'], state_dict[f'layer{li}.attn_wq_lora_a'], state_dict[f'layer{li}.attn_wq_lora_b'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        for h in range(n_head):
            hs = h * head_dim
            for j in range(0, head_dim, 2):
                theta = 10000 ** (-j / head_dim)
                m_theta = pos_id * theta
                cos, sin = math.cos(m_theta), math.sin(m_theta)
                q_r, q_i = q[hs+j], q[hs+j+1]
                q[hs+j], q[hs+j+1] = q_r * cos - q_i * sin, q_r * sin + q_i * cos
                k_r, k_i = k[hs+j], k[hs+j+1]
                k[hs+j], k[hs+j+1] = k_r * cos - k_i * sin, k_r * sin + k_i * cos
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h, k_h = q[hs:hs+head_dim], [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_weights = softmax([sum((q_h[j] * k_h[t][j] for j in range(head_dim)), Value(0)) / head_dim**0.5 for t in range(len(k_h))])
            x_attn.extend([sum((attn_weights[t] * v_h[t][j] for t in range(len(v_h))), Value(0)) for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), x_res)]
        x_res = x
        x = rmsnorm(x)
        gate_weights = softmax(linear(x, state_dict[f'layer{li}.moe_gate']))
        moe_out = [Value(0) for _ in range(n_embd)]
        for e in range(2):
            expert_x = linear(x, state_dict[f'layer{li}.expert{e}.fc1'])
            expert_x = [xi.gelu() for xi in expert_x]
            expert_x = linear(expert_x, state_dict[f'layer{li}.expert{e}.fc2'])
            moe_out = [m + (ex * gate_weights[e]) for m, ex in zip(moe_out, expert_x)]
        x = [a + b for a, b in zip(moe_out, x_res)]
    return linear(x, state_dict['lm_head'])

# --- Training Loop ---
learning_rate, beta1, beta2 = 0.01, 0.85, 0.99
m, v = [0.0] * len(params), [0.0] * len(params)
for step in range(100):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        losses.append(-probs[tokens[pos_id + 1]].log())
    loss = (1 / n) * sum(losses, Value(0))
    loss.backward()
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= learning_rate * m_hat / (v_hat ** 0.5 + 1e-8)
        p.grad = 0
    if step % 10 == 0: print(f"step {step+1} | loss {loss.data:.4f}")
