const byId = (id) => document.getElementById(id);

const formatNumber = (value) => {
  if (!Number.isFinite(value)) {
    return "غير معرّف";
  }
  return Number.isInteger(value) ? value.toString() : value.toFixed(4);
};

const factorial = (n) => {
  if (n < 0) return null;
  let result = 1;
  for (let i = 2; i <= n; i += 1) {
    result *= i;
  }
  return result;
};

const gcd = (a, b) => {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    const temp = y;
    y = x % y;
    x = temp;
  }
  return x;
};

const lcm = (a, b) => {
  if (a === 0 || b === 0) return 0;
  return Math.abs(a * b) / gcd(a, b);
};

const isPrime = (n) => {
  if (n <= 1 || !Number.isInteger(n)) return false;
  if (n <= 3) return true;
  if (n % 2 === 0 || n % 3 === 0) return false;
  for (let i = 5; i * i <= n; i += 6) {
    if (n % i === 0 || n % (i + 2) === 0) return false;
  }
  return true;
};

const solveLinear = () => {
  const a = Number(byId("linear-a").value);
  const b = Number(byId("linear-b").value);
  const resultEl = byId("linear-result");

  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    resultEl.textContent = "يرجى إدخال قيم صحيحة.";
    return;
  }
  if (a === 0) {
    resultEl.textContent = b === 0 ? "المعادلة لها عدد لا نهائي من الحلول." : "لا يوجد حل.";
    return;
  }
  const x = -b / a;
  resultEl.textContent = `الحل: x = ${formatNumber(x)}`;
};

const solveQuadratic = () => {
  const a = Number(byId("quad-a").value);
  const b = Number(byId("quad-b").value);
  const c = Number(byId("quad-c").value);
  const resultEl = byId("quad-result");

  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) {
    resultEl.textContent = "يرجى إدخال جميع القيم.";
    return;
  }
  if (a === 0) {
    resultEl.textContent = "المعادلة ليست تربيعية (a يجب ألا يساوي صفر).";
    return;
  }

  const discriminant = b * b - 4 * a * c;
  if (discriminant > 0) {
    const sqrtD = Math.sqrt(discriminant);
    const x1 = (-b + sqrtD) / (2 * a);
    const x2 = (-b - sqrtD) / (2 * a);
    resultEl.textContent = `جذران حقيقيان: x1 = ${formatNumber(x1)}, x2 = ${formatNumber(x2)}`;
  } else if (discriminant === 0) {
    const x = -b / (2 * a);
    resultEl.textContent = `جذر واحد مكرر: x = ${formatNumber(x)}`;
  } else {
    const real = -b / (2 * a);
    const imag = Math.sqrt(Math.abs(discriminant)) / (2 * a);
    resultEl.textContent = `جذور مركبة: x1 = ${formatNumber(real)} + ${formatNumber(imag)}i, x2 = ${formatNumber(real)} - ${formatNumber(imag)}i`;
  }
};

const calcGcd = () => {
  const a = Number(byId("gcd-a").value);
  const b = Number(byId("gcd-b").value);
  const resultEl = byId("gcd-result");

  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    resultEl.textContent = "يرجى إدخال عددين صحيحين.";
    return;
  }

  const gcdValue = gcd(a, b);
  const lcmValue = lcm(a, b);
  resultEl.textContent = `القاسم المشترك الأكبر = ${gcdValue}، المضاعف المشترك الأصغر = ${lcmValue}`;
};

const checkPrime = () => {
  const value = Number(byId("prime-input").value);
  const resultEl = byId("prime-result");

  if (!Number.isFinite(value)) {
    resultEl.textContent = "يرجى إدخال عدد صحيح.";
    return;
  }

  resultEl.textContent = isPrime(value)
    ? `${value} عدد أولي.`
    : `${value} ليس عددًا أوليًا.`;
};

const calcPerm = () => {
  const n = Number(byId("perm-n").value);
  const r = Number(byId("perm-r").value);
  const resultEl = byId("perm-result");

  if (!Number.isInteger(n) || !Number.isInteger(r) || n < 0 || r < 0 || n < r) {
    resultEl.textContent = "يرجى إدخال قيم صحيحة بحيث n ≥ r ≥ 0.";
    return;
  }

  const nFact = factorial(n);
  const rFact = factorial(r);
  const nMinusRFact = factorial(n - r);

  const permutations = nFact / nMinusRFact;
  const combinations = nFact / (rFact * nMinusRFact);

  resultEl.textContent = `nPr = ${formatNumber(permutations)}، nCr = ${formatNumber(combinations)}`;
};

const calcDet = () => {
  const a = Number(byId("det-a").value);
  const b = Number(byId("det-b").value);
  const c = Number(byId("det-c").value);
  const d = Number(byId("det-d").value);
  const resultEl = byId("det-result");

  if (![a, b, c, d].every((value) => Number.isFinite(value))) {
    resultEl.textContent = "يرجى إدخال جميع قيم المصفوفة.";
    return;
  }

  const det = a * d - b * c;
  resultEl.textContent = `المحدد = ${formatNumber(det)}`;
};

const solveSystem = () => {
  const a1 = Number(byId("sys-a1").value);
  const b1 = Number(byId("sys-b1").value);
  const c1 = Number(byId("sys-c1").value);
  const a2 = Number(byId("sys-a2").value);
  const b2 = Number(byId("sys-b2").value);
  const c2 = Number(byId("sys-c2").value);
  const resultEl = byId("system-result");

  if (![a1, b1, c1, a2, b2, c2].every((value) => Number.isFinite(value))) {
    resultEl.textContent = "يرجى إدخال جميع القيم.";
    return;
  }

  const det = a1 * b2 - a2 * b1;
  if (det === 0) {
    resultEl.textContent = "لا يوجد حل فريد (النظام قد يكون غير متسق أو له حلول متعددة).";
    return;
  }

  const x = (c1 * b2 - c2 * b1) / det;
  const y = (a1 * c2 - a2 * c1) / det;
  resultEl.textContent = `الحل: x = ${formatNumber(x)}، y = ${formatNumber(y)}`;
};

const calcSequence = () => {
  const a1 = Number(byId("seq-a1").value);
  const d = Number(byId("seq-d").value);
  const n = Number(byId("seq-n").value);
  const resultEl = byId("seq-result");

  if (!Number.isFinite(a1) || !Number.isFinite(d) || !Number.isFinite(n) || n <= 0) {
    resultEl.textContent = "يرجى إدخال قيم صحيحة.";
    return;
  }

  const term = a1 + (n - 1) * d;
  resultEl.textContent = `الحد رقم ${n} = ${formatNumber(term)}`;
};

const actions = {
  "solve-linear": solveLinear,
  "solve-quadratic": solveQuadratic,
  "calc-gcd": calcGcd,
  "check-prime": checkPrime,
  "calc-perm": calcPerm,
  "calc-det": calcDet,
  "solve-system": solveSystem,
  "calc-seq": calcSequence,
};

document.querySelectorAll("button[data-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.action;
    actions[action]?.();
  });
});
