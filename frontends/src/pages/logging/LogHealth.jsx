// src/pages/logging/LogHealth.jsx
import React, { useState } from 'react';
import { MealForm } from './MealForm';
import { ExerciseForm } from './ExerciseForm';
import { SleepForm } from './SleepForm';
import { MoodForm } from './MoodForm';

export const LogHealth = () => {
  const [activeLogType, setActiveLogType] = useState('meal');

  return (
    <div className="main-content">
      <div className="logging-container">
        <h2>Log Health Data</h2>
        
        <div className="log-type-tabs">
          <button 
            className={activeLogType === 'meal' ? 'active' : ''}
            onClick={() => setActiveLogType('meal')}
          >
            🍽️ Meal
          </button>
          <button 
            className={activeLogType === 'exercise' ? 'active' : ''}
            onClick={() => setActiveLogType('exercise')}
          >
            🏃‍♂️ Exercise
          </button>
          <button 
            className={activeLogType === 'sleep' ? 'active' : ''}
            onClick={() => setActiveLogType('sleep')}
          >
            😴 Sleep
          </button>
          <button 
            className={activeLogType === 'mood' ? 'active' : ''}
            onClick={() => setActiveLogType('mood')}
          >
            😊 Mood
          </button>
        </div>
        
        <div className="log-form-container">
          {activeLogType === 'meal' && <MealForm />}
          {activeLogType === 'exercise' && <ExerciseForm />}
          {activeLogType === 'sleep' && <SleepForm />}
          {activeLogType === 'mood' && <MoodForm />}
        </div>
      </div>
    </div>
  );
};