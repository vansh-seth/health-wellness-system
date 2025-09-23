// src/pages/auth/Signup.jsx
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const Signup = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    first_name: '',
    last_name: '',
    age: '',
    gender: 'male',
    height: '',
    weight: '',
    activity_level: 'moderately_active',
    health_goals: ['maintain_health'],
    medical_conditions: [],
    allergies: []
  });
  const [error, setError] = useState('');
  
  const { register, loading } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validate form
    if (formData.age < 1 || formData.age > 120) {
      setError('Please enter a valid age');
      return;
    }
    if (formData.height < 50 || formData.height > 250) {
      setError('Please enter a valid height (50-250 cm)');
      return;
    }
    if (formData.weight < 20 || formData.weight > 300) {
      setError('Please enter a valid weight (20-300 kg)');
      return;
    }

    const userData = {
      ...formData,
      age: parseInt(formData.age),
      height: parseFloat(formData.height),
      weight: parseFloat(formData.weight)
    };

    const result = await register(userData);
    
    if (result.success) {
      navigate('/dashboard');
    } else {
      setError(result.error);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form signup-form">
        <h2>Create Account</h2>
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <input
              type="text"
              name="first_name"
              placeholder="First Name"
              value={formData.first_name}
              onChange={handleChange}
              required
            />
            <input
              type="text"
              name="last_name"
              placeholder="Last Name"
              value={formData.last_name}
              onChange={handleChange}
              required
            />
          </div>

          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            required
          />
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            required
          />

          <div className="form-row">
            <input
              type="number"
              name="age"
              placeholder="Age"
              value={formData.age}
              onChange={handleChange}
              min="1"
              max="120"
              required
            />
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="form-row">
            <input
              type="number"
              name="height"
              placeholder="Height (cm)"
              value={formData.height}
              onChange={handleChange}
              min="50"
              max="250"
              step="0.1"
              required
            />
            <input
              type="number"
              name="weight"
              placeholder="Weight (kg)"
              value={formData.weight}
              onChange={handleChange}
              min="20"
              max="300"
              step="0.1"
              required
            />
          </div>

          <label>Activity Level:</label>
          <select name="activity_level" value={formData.activity_level} onChange={handleChange}>
            <option value="sedentary">Sedentary (little to no exercise)</option>
            <option value="lightly_active">Lightly Active (light exercise 1-3 days/week)</option>
            <option value="moderately_active">Moderately Active (moderate exercise 3-5 days/week)</option>
            <option value="very_active">Very Active (hard exercise 6-7 days/week)</option>
          </select>
          
          <button type="submit" disabled={loading}>
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>
        
        <p className="auth-switch">
          Already have an account? <Link to="/login">Login here</Link>
        </p>
      </div>
    </div>
  );
};

export default Signup;