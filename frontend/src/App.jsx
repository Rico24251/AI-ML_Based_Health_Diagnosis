import { useState } from 'react'
import DiseaseGrid from './components/DiseaseGrid'
import DiagnosticModal from './components/DiagnosticModal'

function App() {
  const [activeDisease, setActiveDisease] = useState(null)

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center py-12">
      
      {/* Dashboard Header - Only show if NO disease is selected */}
      {!activeDisease && (
        <div className="text-center mb-12 px-4">
          <h1 className="text-5xl font-extrabold text-slate-900 mb-4 tracking-tight">Health AI Assistant</h1>
          <p className="text-lg text-slate-500 max-w-2xl mx-auto">
            Select a diagnostic module below to begin analyzing patient data, X-rays, or audio samples using our advanced machine learning models.
          </p>
        </div>
      )}

      {/* The Routing Logic */}
      {activeDisease ? (
        <DiagnosticModal 
          disease={activeDisease} 
          onClose={() => setActiveDisease(null)} 
        />
      ) : (
        <DiseaseGrid onSelectDisease={setActiveDisease} />
      )}

    </div>
  )
}

export default App