import React, { useState, useEffect } from 'react';

const VisualisasiWarnaPiksel = () => {
  const [merah, setMerah] = useState(128);
  const [hijau, setHijau] = useState(128);
  const [biru, setBiru] = useState(128);
  const [lebarGambar, setLebarGambar] = useState(1920);
  const [tinggiGambar, setTinggiGambar] = useState(1080);

  const saluranKeHex = (saluran) => {
    const hex = saluran.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };

  const rgbKeHex = (r, g, b) => {
    return "#" + saluranKeHex(r) + saluranKeHex(g) + saluranKeHex(b);
  };

  const hitungUkuranGambar = () => {
    const bitPerPiksel = 3 * 8;
    const totalPiksel = lebarGambar * tinggiGambar;
    const totalBit = totalPiksel * bitPerPiksel;
    const totalByte = totalBit / 8;
    const totalKilobyte = totalByte / 1024;
    const totalMegabyte = totalKilobyte / 1024;
    return totalMegabyte.toFixed(2);
  };

  const generateRandomColor = () => {
    return {
      r: Math.floor(Math.random() * 256),
      g: Math.floor(Math.random() * 256),
      b: Math.floor(Math.random() * 256)
    };
  };

  const renderPixelGrid = () => {
    const gridSize = 10;
    const pixels = [];
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const isCenter = i === Math.floor(gridSize/2) && j === Math.floor(gridSize/2);
        const color = isCenter ? rgbKeHex(merah, hijau, biru) : rgbKeHex(generateRandomColor().r, generateRandomColor().g, generateRandomColor().b);
        pixels.push(
          <div
            key={`${i}-${j}`}
            className="w-4 h-4 border border-gray-200"
            style={{ backgroundColor: color }}
          ></div>
        );
      }
    }
    return pixels;
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">Visualisasi Warna Piksel dan Ukuran Gambar</h2>
      
      <div className="flex mb-6">
        <div className="w-1/2 pr-4">
          <h3 className="text-lg font-semibold mb-2">Saluran Warna</h3>
          <div className="mb-4">
            <label className="block mb-1">Merah: {merah}</label>
            <input
              type="range"
              min="0"
              max="255"
              value={merah}
              onChange={(e) => setMerah(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="mb-4">
            <label className="block mb-1">Hijau: {hijau}</label>
            <input
              type="range"
              min="0"
              max="255"
              value={hijau}
              onChange={(e) => setHijau(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="mb-4">
            <label className="block mb-1">Biru: {biru}</label>
            <input
              type="range"
              min="0"
              max="255"
              value={biru}
              onChange={(e) => setBiru(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
        <div className="w-1/2 pl-4">
          <h3 className="text-lg font-semibold mb-2">Warna yang Dihasilkan</h3>
          <div
            className="w-full h-32 border border-gray-300"
            style={{ backgroundColor: rgbKeHex(merah, hijau, biru) }}
          ></div>
          <p className="mt-2">Hex: {rgbKeHex(merah, hijau, biru)}</p>
          <p>RGB: ({merah}, {hijau}, {biru})</p>
        </div>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Ilustrasi Piksel dalam Gambar</h3>
        <div className="flex items-center">
          <div className="w-40 h-40 border border-gray-300 grid grid-cols-10">
            {renderPixelGrid()}
          </div>
          <div className="ml-4 w-40 h-40 border-4 border-gray-500 rounded-full flex items-center justify-center overflow-hidden relative">
            <div
              className="w-80 h-80 grid grid-cols-10"
              style={{
                transform: 'scale(2)',
                transformOrigin: 'center'
              }}
            >
              {renderPixelGrid()}
            </div>
          </div>
        </div>
        <p className="mt-2">Piksel yang diperbesar ditunjukkan di tengah grid.</p>
      </div>

      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Perhitungan Ukuran Gambar</h3>
        <div className="flex mb-4">
          <div className="w-1/2 pr-2">
            <label className="block mb-1">Lebar Gambar (piksel)</label>
            <input
              type="number"
              value={lebarGambar}
              onChange={(e) => setLebarGambar(Number(e.target.value))}
              className="w-full p-1 border border-gray-300 rounded"
            />
          </div>
          <div className="w-1/2 pl-2">
            <label className="block mb-1">Tinggi Gambar (piksel)</label>
            <input
              type="number"
              value={tinggiGambar}
              onChange={(e) => setTinggiGambar(Number(e.target.value))}
              className="w-full p-1 border border-gray-300 rounded"
            />
          </div>
        </div>
        <p>Total Piksel: {lebarGambar * tinggiGambar}</p>
        <p>Bit per Piksel: 24 (8 bit per saluran × 3 saluran)</p>
        <p>Ukuran Gambar Tanpa Kompresi: {hitungUkuranGambar()} MB</p>
      </div>

      <div className="bg-gray-100 p-4 rounded">
        <h3 className="text-lg font-semibold mb-2">Cara Kerjanya</h3>
        <p>Setiap piksel dalam gambar digital biasanya direpresentasikan oleh tiga saluran warna: Merah, Hijau, dan Biru (RGB).</p>
        <p>Setiap saluran menggunakan 8 bit, memungkinkan 256 tingkat intensitas yang berbeda (0-255) per saluran.</p>
        <p>Kombinasi dari ketiga saluran ini menciptakan warna akhir dari setiap piksel.</p>
        <p>Dengan 8 bit per saluran, kita memiliki 256 × 256 × 256 = 16.777.216 kemungkinan warna.</p>
        <p>Total jumlah bit yang diperlukan untuk merepresentasikan sebuah gambar adalah:</p>
        <p className="font-mono">Lebar Gambar × Tinggi Gambar × 3 saluran × 8 bit per saluran</p>
        <p>Ini menentukan ukuran file gambar tanpa kompresi.</p>
      </div>
    </div>
  );
};

export default VisualisasiWarnaPiksel;