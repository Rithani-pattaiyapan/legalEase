import express from 'express';
import LawyerProfile from '../module/lawyer.module.js';
import Case from '../module/case.module.js';

const router = express.Router();

router.get('/', (req, res) => {
    res.send('Lawyer Router is working');
});

// Filter lawyers based on specialization, location, experience_years
router.post('/filter',async(req,res)=>{
    try {
        const {specialization, location, experience_years} = req.body;
        // Implement filtering logic here
        const filteredLawyers = await LawyerProfile.find({
            ...(specialization && { specialization }),
            ...(location && { location :  { $regex: location, $options: 'i' } }),
            ...(experience_years && { experience_years: { $gte: experience_years } })
        })
        .populate('user_id', 'name email phone')
        .select('specialization experience_years location availability ratings bar_council_id approved_status');

        res.status(200).json({
            success : true,
            lawyers: filteredLawyers
        });
    }
    catch (error) {
        res.status(500).json({
            success : false,
            message: "Server Error", 
            error: error.message
        });
    }
});

// Get all cases for a specific client
router.get('/:client_id/cases', async (req, res) => {
    try {
        const { client_id } = req.params;
        const cases = await Case.find({ client_id }).populate('lawyer_id', '_id name specialization location experience_years');
        res.status(200).json({
            success: true,
            cases
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: "Server Error",
            error: error.message
        });
    }
});

// Get case details along with client information
router.get('/:case_id/clients', async (req, res) => {
    try {
        const { case_id } = req.params;
        const caseData = await Case.findById(case_id).populate('client_id', 'title description legal_category status process_updates created_at updated_at');
        if (!caseData) {
            return res.status(404).json({
                success: false,
                message: "Case not found"
            });
        }
        res.status(200).json({
            success: true,
            caseData
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: "Server Error",
            error: error.message
        }); 
    }
});


export default router;