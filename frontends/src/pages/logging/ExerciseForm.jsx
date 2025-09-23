// pages/logging/ExerciseForm.jsx
import React, { useState } from 'react';
import { api } from '../../services/api';

export const ExerciseForm = () => {
  const [formData, setFormData] = useState({
    exercise_name: '',
    duration_minutes: '',
    calories_burned: '',
    exercise_type: 'cardio'
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setSuccess(false);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const exerciseData = {
        ...formData,
        duration_minutes: parseInt(formData.duration_minutes),
        calories_burned: parseInt(formData.calories_burned) || 0
      };

      await api.logExercise(exerciseData);
      setSuccess(true);
      setFormData({
        exercise_name: '',
        duration_minutes: '',
        calories_burned: '',
        exercise_type: 'cardio'
      });
    } catch (error) {
      console.error('Error logging exercise:', error);
      setError(error.message || 'Failed to log exercise');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-wrapper">
      <h3>Log Exercise</h3>
      {success && <div className="success-message">Exercise logged successfully!</div>}
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit} className="log-form">
        <input
          type="text"
          name="exercise_name"
          placeholder="Exercise Name (e.g., Morning Run)"
          value={formData.exercise_name}
          onChange={handleChange}
          required
        />
        
        <input
          type="number"
          name="duration_minutes"
          placeholder="Duration (minutes)"
          value={formData.duration_minutes}
          onChange={handleChange}
          min="1"
          required
        />
        
        <input
          type="number"
          name="calories_burned"
          placeholder="Calories Burned (optional)"
          value={formData.calories_burned}
          onChange={handleChange}
          min="0"
        />
        
        <select name="exercise_type" value={formData.exercise_type} onChange={handleChange}>
          <option value="cardio">Cardio</option>
          <option value="strength">Strength Training</option>
          <option value="flexibility">Flexibility/Stretching</option>
          <option value="sports">Sports</option>
        </select>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Logging Exercise...' : 'Log Exercise'}
        </button>
      </form>
    </div>
  );
};