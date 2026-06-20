const CACHE_NAME = 'agrivision-v6';
const ASSETS = [
    './',
    './index.html',
    './script.js',
    './manifest.json',
    './rost-logo.png',
    './icon-192.png',
    './icon-512.png',
    './icon-maskable-512.png',
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

self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then((keys) =>
            Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
        )
    );
});

self.addEventListener('fetch', (e) => {
    // ignoreSearch: serve dalla cache anche le URL con ?v=... (es. script.js?v=40)
    e.respondWith(
        caches.match(e.request, { ignoreSearch: true }).then((response) => response || fetch(e.request))
    );
});
