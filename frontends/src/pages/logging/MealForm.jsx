// pages/logging/MealForm.jsx
import React, { useState } from 'react';
import { api } from '../../services/api';

export const MealForm = () => {
  const [formData, setFormData] = useState({
    meal_name: '',
    calories: '',
    protein: '',
    carbs: '',
    fat: '',
    meal_type: 'breakfast'
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
      const mealData = {
        ...formData,
        calories: parseInt(formData.calories),
        protein: parseFloat(formData.protein) || 0,
        carbs: parseFloat(formData.carbs) || 0,
        fat: parseFloat(formData.fat) || 0
      };

      await api.logMeal(mealData);
      setSuccess(true);
      setFormData({
        meal_name: '',
        calories: '',
        protein: '',
        carbs: '',
        fat: '',
        meal_type: 'breakfast'
      });
    } catch (error) {
      console.error('Error logging meal:', error);
      setError(error.message || 'Failed to log meal');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-wrapper">
      <h3>Log Meal</h3>
      {success && <div className="success-message">Meal logged successfully!</div>}
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit} className="log-form">
        <input
          type="text"
          name="meal_name"
          placeholder="Meal Name (e.g., Chicken Salad)"
          value={formData.meal_name}
          onChange={handleChange}
          required
        />
        
        <input
          type="number"
          name="calories"
          placeholder="Calories"
          value={formData.calories}
          onChange={handleChange}
          min="1"
          required
        />
        
        <div className="form-row">
          <input
            type="number"
            step="0.1"
            name="protein"
            placeholder="Protein (g)"
            value={formData.protein}
            onChange={handleChange}
            min="0"
          />
          <input
            type="number"
            step="0.1"
            name="carbs"
            placeholder="Carbs (g)"
            value={formData.carbs}
            onChange={handleChange}
            min="0"
          />
          <input
            type="number"
            step="0.1"
            name="fat"
            placeholder="Fat (g)"
            value={formData.fat}
            onChange={handleChange}
            min="0"
          />
        </div>
        
        <select name="meal_type" value={formData.meal_type} onChange={handleChange}>
          <option value="breakfast">Breakfast</option>
          <option value="lunch">Lunch</option>
          <option value="dinner">Dinner</option>
          <option value="snack">Snack</option>
        </select>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Logging Meal...' : 'Log Meal'}
        </button>
      </form>
    </div>
  );
};