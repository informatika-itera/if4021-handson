import React from 'react';
import ADCVisualization from './components/visualizations/ADCVisualization';
import JPEGCompressionVisualization from './components/visualizations/JPEGCompressionVisualization';

// Simple Separator component
const Separator = () => <hr className="my-8 border-t border-gray-300" />;

function App() {
  return (
    <div className="App p-4">
      <h1 className="text-3xl font-bold mb-8">IF4021 Visualizations</h1>
      
      <ADCVisualization />
      
      <Separator />
      
      <JPEGCompressionVisualization />
    </div>
  );
}

export default App;