const CACHE_NAME = 'agrivision-v4';
const ASSETS = [
    './',
    './index.html',
    './script.js',
    './manifest.json',
    './libs/tf.min.js',
    './libs/tf-tflite.min.js',
    './libs/tflite_web_api_cc.wasm',
    './libs/tflite_web_api_cc_simd.wasm',
    './libs/tflite_web_api_cc_simd.js',
    './libs/tflite_web_api_cc.js',
    './assets/models/best_float32.tflite'
];

self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', (e) => {
    e.respondWith(
        caches.match(e.request).then((response) => response || fetch(e.request))
    );
});
